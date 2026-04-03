FROM python:3.11-slim

WORKDIR /app

# Install dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (cached layer — ~90MB, only downloads once)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy pipeline code
COPY code/ code/

# Copy data (includes companies.jsonl, queries.json, and cache/)
COPY data/ data/

# Copy taxonomy dictionaries
COPY taxonomy/ taxonomy/

# Copy entry points and UI
COPY app.py .
COPY solution.py .
COPY run.py .
COPY frontend.html .

# Create training directory for auto-logged pairs
RUN mkdir -p training

EXPOSE 8000

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]