"""Retrain the land-cover classifier and register it in MLflow (phase 2).

Skeleton DAG: the tasks are wired to the stack (training database, MLflow)
and document what each step must do. Replace the bodies with the extraction,
training and HPO logic validated in the phase 1 notebooks.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from airflow.sdk import dag, task

log = logging.getLogger(__name__)


@dag(
    dag_id="retrain_model",
    description="Retrain the Sentinel-2 land-cover classifier and register the best run",
    schedule=None,  # manual trigger; set a cron once the pipeline is validated
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sentinel2", "phase2"],
)
def retrain_model():
    @task
    def load_samples() -> int:
        """Count the pixel samples available in training.samples."""
        import psycopg2

        with psycopg2.connect(
            host=os.environ["TRAIN_DB_HOST"],
            port=os.environ["TRAIN_DB_PORT"],
            dbname=os.environ["TRAIN_DB_NAME"],
            user=os.environ["TRAIN_DB_USER"],
            password=os.environ["TRAIN_DB_PASSWORD"],
        ) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM training.samples")
            (count,) = cur.fetchone()
        log.info("training.samples rows: %s", count)
        return int(count)

    @task
    def train_candidates(sample_count: int) -> str:
        """Train the candidate models with HPO and log them as MLflow runs.

        TODO: read the samples, split train/validation/test, run the six
        architectures with their tuned hyper-parameters, log params, metrics
        (overall accuracy, kappa, precision, recall, F1) and the confusion
        matrices, and return the run id of the best candidate.
        """
        import mlflow

        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        with mlflow.start_run(run_name="retrain-skeleton") as run:
            mlflow.log_param("sample_count", sample_count)
            mlflow.log_param("band_names", os.environ["BAND_NAMES"])
            log.info("run %s logged; training not implemented yet", run.info.run_id)
            return run.info.run_id

    @task
    def register_best(mlflow_run_id: str) -> None:
        """Register the best run under MODEL_NAME and move MODEL_ALIAS to it.

        TODO: compare the candidate with the current production version on the
        same validation set and promote only when it improves. (`run_id` is a
        reserved Airflow context kwarg, hence the parameter name.)
        """
        log.info(
            "would register run %s as models:/%s@%s",
            mlflow_run_id, os.environ["MODEL_NAME"], os.environ["MODEL_ALIAS"],
        )

    register_best(train_candidates(load_samples()))


retrain_model()
