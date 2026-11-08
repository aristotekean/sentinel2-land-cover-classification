# Airflow (phase 2) image: official base plus the Python dependencies the DAGs
# need to read the training database and the bands, retrain the model,
# register it in MLflow and write the predicted GeoTIFF.
#
# Build context is the repository root, consistent with the other services:
#   docker compose build airflow-scheduler
#   docker build -f apps/airflow/Dockerfile -t sentinel2-airflow .

ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}
ARG AIRFLOW_VERSION

USER root
COPY apps/airflow/requirements.txt /tmp/requirements.txt
# DAGs are baked in so the image is self-contained; the local compose still
# mounts apps/airflow/dags over this copy for fast iteration.
COPY apps/airflow/dags /opt/airflow/dags
RUN chown -R airflow:root /tmp/requirements.txt /opt/airflow/dags
USER airflow

# Re-stating apache-airflow pins its version so pip cannot upgrade or
# downgrade it while resolving the DAG dependencies.
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements.txt
