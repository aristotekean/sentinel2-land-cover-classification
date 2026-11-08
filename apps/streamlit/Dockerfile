# Streamlit (phase 2) image: dashboard with the map, the metrics of the
# production model and the class areas.
#
# Build context is the repository root, consistent with the other services:
#   docker compose build streamlit
#   docker build -f apps/streamlit/Dockerfile -t sentinel2-streamlit .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app
COPY apps/streamlit/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/streamlit/src/ /app/src/
RUN chown -R 1000:1000 /app

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
