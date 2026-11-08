# FastAPI (optional) image: serves models:/<MODEL_NAME>@<MODEL_ALIAS> for
# single-pixel inference.
#
# Build context is the repository root, consistent with the other services:
#   docker compose build api
#   docker build -f apps/api/Dockerfile -t sentinel2-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src"

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app
COPY apps/api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/src/ /app/src/
RUN chown -R 1000:1000 /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
