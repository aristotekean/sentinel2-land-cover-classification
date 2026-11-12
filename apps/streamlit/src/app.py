"""Sentinel-2 land cover dashboard (phase 2): map, metrics and class areas.

Reads the production model from the MLflow registry, the predicted GeoTIFFs
from the MinIO data bucket and the class areas from the training database.
Everything is configured through the environment (see docker-compose.yml).
"""

from __future__ import annotations

import os

import boto3
import pandas as pd
import sqlalchemy
import streamlit as st
from mlflow import MlflowClient

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]
DATA_BUCKET = os.environ["DATA_BUCKET"]
PREDICTIONS_PREFIX = "predictions/"


@st.cache_resource
def mlflow_client() -> MlflowClient:
    return MlflowClient()


@st.cache_resource
def s3_client():
    return boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])


@st.cache_resource
def train_db() -> sqlalchemy.Engine:
    url = sqlalchemy.URL.create(
        "postgresql+psycopg2",
        username=os.environ["TRAIN_DB_USER"],
        password=os.environ["TRAIN_DB_PASSWORD"],
        host=os.environ["TRAIN_DB_HOST"],
        port=int(os.environ["TRAIN_DB_PORT"]),
        database=os.environ["TRAIN_DB_NAME"],
    )
    return sqlalchemy.create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def production_model() -> dict | None:
    version = mlflow_client().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = mlflow_client().get_run(version.run_id)
    return {
        "name": version.name,
        "version": version.version,
        "run_id": version.run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
    }


@st.cache_data(ttl=60)
def predictions() -> list[str]:
    response = s3_client().list_objects_v2(Bucket=DATA_BUCKET, Prefix=PREDICTIONS_PREFIX)
    return [
        item["Key"]
        for item in response.get("Contents", [])
        if item["Key"].lower().endswith((".tif", ".tiff", ".jp2"))
    ]


@st.cache_data(ttl=60)
def class_areas(prediction: str) -> pd.DataFrame:
    query = sqlalchemy.text(
        "SELECT class_id, pixel_count, area_ha FROM training.class_areas "
        "WHERE prediction = :prediction ORDER BY class_id"
    )
    with train_db().connect() as conn:
        return pd.read_sql(query, conn, params={"prediction": prediction})


st.set_page_config(page_title="Sentinel-2 land cover", page_icon="🛰️", layout="wide")
st.title("Sentinel-2 land cover · Bogotá (T18NWL)")

with st.sidebar:
    st.header("Production model")
    try:
        model = production_model()
        st.success(f"{model['name']} v{model['version']} ({MODEL_ALIAS})")
    except Exception as exc:  # registry empty or MLflow unreachable
        model = None
        st.warning(f"No model with alias `{MODEL_ALIAS}` yet.\n\n{exc}")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

map_tab, metrics_tab, areas_tab = st.tabs(["Map", "Metrics", "Areas"])

with map_tab:
    st.subheader("Predicted land-cover raster")
    try:
        keys = predictions()
    except Exception as exc:
        keys = []
        st.warning(f"Cannot list s3://{DATA_BUCKET}/{PREDICTIONS_PREFIX}: {exc}")
    if not keys:
        st.info("No predicted raster in the data bucket yet. Run the `predict_raster` DAG.")
    else:
        selected = st.selectbox("Prediction", keys)
        st.caption(f"s3://{DATA_BUCKET}/{selected}")
        # TODO: render the GeoTIFF with a discrete colour map (one colour per class).
        st.info("Raster rendering is not implemented yet.")

with metrics_tab:
    st.subheader("Validation metrics of the production model")
    if model is None:
        st.info("Metrics appear once a model carries the production alias.")
    else:
        metrics = pd.DataFrame(
            sorted(model["metrics"].items()), columns=["metric", "value"]
        )
        st.dataframe(metrics, hide_index=True, width="stretch")
        with st.expander("Parameters"):
            st.json(model["params"])

with areas_tab:
    st.subheader("Area per class")
    if not keys:
        st.info("Areas are computed from a predicted raster.")
    else:
        try:
            areas = class_areas(selected)
        except Exception as exc:
            areas = pd.DataFrame()
            st.warning(f"Cannot read training.class_areas: {exc}")
        if areas.empty:
            st.info("No areas stored for this prediction yet.")
        else:
            st.dataframe(areas, hide_index=True, width="stretch")
            st.bar_chart(areas.set_index("class_id")["area_ha"])
