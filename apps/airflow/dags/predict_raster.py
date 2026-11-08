"""Full-scene inference: classify every pixel of the scene with the production
model and export a georeferenced GeoTIFF (phase 2).

Skeleton DAG: the tasks are wired to the stack (MLflow registry, MinIO data
bucket, training database) and document what each step must do.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from airflow.sdk import dag, task

log = logging.getLogger(__name__)


@dag(
    dag_id="predict_raster",
    description="Classify the whole Sentinel-2 scene with the production model",
    schedule=None,  # manual trigger, typically after retrain_model
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sentinel2", "phase2"],
)
def predict_raster():
    @task
    def resolve_model() -> dict:
        """Resolve models:/<MODEL_NAME>@<MODEL_ALIAS> in the MLflow registry."""
        from mlflow import MlflowClient

        name, alias = os.environ["MODEL_NAME"], os.environ["MODEL_ALIAS"]
        version = MlflowClient().get_model_version_by_alias(name, alias)
        log.info("resolved %s@%s -> version %s", name, alias, version.version)
        return {"name": name, "alias": alias, "version": version.version}

    @task
    def predict_scene(model: dict) -> str:
        """Run the model over the ten bands and write the predicted raster.

        TODO: read the bands from s3://<DATA_BUCKET>/bands/, stack them in the
        BAND_NAMES order, predict every pixel, keep the original CRS and
        transform, and upload the single-band GeoTIFF to
        s3://<DATA_BUCKET>/predictions/. Return the object key.
        """
        key = f"predictions/{model['name']}-v{model['version']}.tif"
        log.info("would write %s to bucket %s", key, os.environ["DATA_BUCKET"])
        return key

    @task
    def compute_class_areas(prediction_key: str) -> None:
        """Count the pixels of each class and store the area in hectares.

        TODO: read the predicted raster, count pixels per class, convert with
        the pixel size (20 m -> 0.04 ha) and upsert training.class_areas.
        """
        log.info("would compute class areas of %s", prediction_key)

    compute_class_areas(predict_scene(resolve_model()))


predict_raster()
