# MLflow tracking server image.
#
# The official image ships MLflow only; the server needs two client libraries
# for the stack: psycopg2 (Postgres backend store) and boto3 (MinIO artifacts).
# Baking them in keeps starts reproducible and offline.
#   docker compose build mlflow
#   docker build -f apps/mlflow/Dockerfile -t sentinel2-mlflow .
FROM ghcr.io/mlflow/mlflow:v3.18.0

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.13 \
    boto3==1.43.126
