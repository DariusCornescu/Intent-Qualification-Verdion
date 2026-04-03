FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time (~90MB)
# so first query doesn't wait 30s for download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY code/ code/
COPY data/ data/
COPY taxonomy/ taxonomy/
COPY app.py .
COPY solution.py .
COPY run.py .
COPY frontend.html .
COPY queries.json .

RUN mkdir -p cache training

# Pre-cached LLM responses — enables demo mode without API key
COPY cache/llm_cache.json cache/llm_cache.json

EXPOSE 8000
CMD ["python", "app.py", "--host", "0.0.0.0"]