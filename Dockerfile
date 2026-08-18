# Keep in sync with CI (.github/workflows/ci.yml): the pinned numpy>=2.5
# requires Python >= 3.12.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_AGENT_VECTOR_BACKEND=memory \
    AI_AGENT_EMBEDDING_PROVIDER=sentence_transformers \
    AI_AGENT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
    HF_HUB_DISABLE_XET=1 \
    AI_AGENT_RERANKER=heuristic

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Cache the real embedding model in the image so startup does not depend on a
# network call. Override AI_AGENT_EMBEDDING_MODEL when building a custom image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

COPY app ./app
COPY data ./data
COPY web ./web

EXPOSE 8015

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8015/api/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8015"]
