"""
ScoringIndex is built once at startup and reused across all queries. It eliminates the previous
bottleneck: re-encoding company texts through the embedding model on every query.

    > Without this index: encoding 477 company texts + query
    > With this index: encode query only, lookup cached vectors 

Disk cache: embedding matrix will be saved to disk after the first build
On subsequent startups, if data hasn't changed (same text hash), embeddings load from disk in a much faster time.

Memory at current scale (477 companies, MiniLM 384-dim):
  - Embedding matrix: ~730KB (477 × 384 × 4 bytes float32)
  - BM25 index: ~2MB (tokenized corpus + IDF weights)

As for the scaling part..at 100k or even more companies we'll switch to a vector database (FAISS IndexHNSW or Qdrant)
for O(log n) Approximate Nearest Neighbor search
"""
import os
import logging
import hashlib
import numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi
from code.models import CompanyProfile

logger = logging.getLogger(__name__)

_embedding_model = None
_EMBEDDING_AVAILABLE = None  
_BM25_AVAILABLE = None

def _get_embedding_model():
    """Lazy-load the sentence-transformer model. Cached after first call."""

    global _embedding_model, _EMBEDDING_AVAILABLE

    if _EMBEDDING_AVAILABLE is False: return None
    if _embedding_model is not None: return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDING_AVAILABLE = True
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        return _embedding_model
    
    except ImportError:
        logger.info("sentence-transformers not installed - embedding signal disabled")
        _EMBEDDING_AVAILABLE = False
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        _EMBEDDING_AVAILABLE = False

        return None

def warm_up_embedding_model():
    """Pre-load the embedding model so the first query doesn't pay the cost."""
    _get_embedding_model()

def _check_bm25():

    global _BM25_AVAILABLE
    if _BM25_AVAILABLE is not None: return _BM25_AVAILABLE
    try:
        _BM25_AVAILABLE = True

    except ImportError:
        logger.info("rank-bm25 not installed - make sure you've installed the requirements")
        _BM25_AVAILABLE = False

    return _BM25_AVAILABLE


class ScoringIndex:
    """ Pre-computed indexes for Stage 2 scoring. Built once at startup, reused across all queries """

    def __init__(self, companies: list[CompanyProfile], cache_dir: str = None):
        self.n = len(companies)
        self._id_to_idx = {id(c): i for i, c in enumerate(companies)}

        # Pre-compute text blobs (shared by BM25 and embeddings)
        self.texts = [c.get_text_blob() or "" for c in companies]
        self._non_empty = [bool(t.strip()) for t in self.texts]

        # Cache directory for persistent embedding storage
        self._cache_dir = cache_dir
        self._data_hash = self._compute_data_hash()

        # Pre-computed embedding matrix (with disk cache)
        self.embedding_matrix = self._load_or_build_embeddings()

        # Pre-build BM25 index on full corpus
        self.bm25_index = self._build_bm25()

        emb_status = f"{self.embedding_matrix.shape}" if self.embedding_matrix is not None else "disabled"
        bm25_status = "ready" if self.bm25_index is not None else "disabled"
        logger.info( f"ScoringIndex built: {self.n} companies, embeddings={emb_status}, bm25={bm25_status}" )

    def _compute_data_hash(self) -> str:
        """Hash all company texts to detect data changes."""

        hash = hashlib.sha256()
        for text in self.texts:
            hash.update(text.encode("utf-8", errors="replace"))
        return hash.hexdigest()[:16]

    def _cache_path(self) -> Optional[str]:

        if not self._cache_dir:
            return None
        
        os.makedirs(self._cache_dir, exist_ok=True)
        return os.path.join(self._cache_dir, f"embeddings_{self._data_hash}.npy")

    def _load_or_build_embeddings(self) -> Optional[np.ndarray]:

        cache_file = self._cache_path()
        if cache_file:
            try:
                if os.path.exists(cache_file):
                    matrix = np.load(cache_file)
                    if matrix.shape[0] == self.n:
                        logger.info(f"Loaded embeddings from cache: {cache_file}")
                        return matrix
                    else:
                        logger.warning(f"Cache shape mismatch ({matrix.shape[0]} vs {self.n}), rebuilding")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, rebuilding")

        matrix = self._build_embeddings()

        if matrix is not None and cache_file:
            try:
                np.save(cache_file, matrix)
                size_kb = matrix.nbytes / 1024
                logger.info(f"Saved embeddings to cache: {cache_file} ({size_kb:.0f} KB)")                
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")

        return matrix

    def _build_embeddings(self) -> Optional[np.ndarray]:
        """Encode all company texts once. Returns (n, dim) matrix or None."""

        model = _get_embedding_model()
        if model is None: return None
        
        texts_to_encode = [text for text, m in zip(self.texts, self._non_empty) if m]
        if not texts_to_encode: return None

        try:
            embeddings = model.encode(texts_to_encode, show_progress_bar=False, batch_size=64)
            full_matrix = np.zeros((self.n, embeddings.shape[1]), dtype=np.float32)
            encoding_idx = 0
            for i, has_text in enumerate(self._non_empty):
                if has_text:
                    full_matrix[i] = embeddings[encoding_idx]
                    encoding_idx += 1

            return full_matrix

        except Exception as e:
            logger.warning(f"Failed to build embedding index: {e}")
            return None

    def _build_bm25(self):
        """Pre-tokenize corpus and build BM25 index."""

        if not _check_bm25(): return None

        try:

            tokenized = [text.lower().split() for text in self.texts]
            if not any(tokenized): return None

            return BM25Okapi(tokenized)
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            return None

    def get_indices(self, filtered_companies: list[CompanyProfile]) -> list[int]:
        """Map filtered company objects to their indices in the full index."""

        indices = []
        for c in filtered_companies:
            idx = self._id_to_idx.get(id(c))
            if idx is not None:
                indices.append(idx)
        return indices


def build_scoring_index(companies: list[CompanyProfile], cache_dir: str = None) -> ScoringIndex:
    """Build the scoring index at startup. Call once after loading companies."""
    return ScoringIndex(companies, cache_dir=cache_dir)