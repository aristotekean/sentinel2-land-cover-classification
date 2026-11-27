"""Sentinel-2 land cover dashboard (phase 2): map, metrics and class areas.

Reads the production model from the MLflow registry, the predicted GeoTIFFs
from the MinIO data bucket and the class areas from the training database.
Everything is configured through the environment (see docker-compose.yml).
"""

from __future__ import annotations

import html
import json
import os

import boto3
import numpy as np
import pandas as pd
import sqlalchemy
import streamlit as st
from mlflow import MlflowClient
from rasterio.io import MemoryFile

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]
DATA_BUCKET = os.environ["DATA_BUCKET"]
PREDICTIONS_PREFIX = "predictions/"
MAP_MAX_PIXELS = 1400  # longest side of the displayed map; the GeoTIFF keeps its resolution


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


@st.cache_data(ttl=600, show_spinner="Rendering the raster…")
def land_cover_map(prediction: str) -> dict:
    """Read a predicted raster and colour it with the table stored in the file.

    The classes are read at full resolution, so the legend lists every class of
    the raster; only the displayed image is reduced.
    """
    body = s3_client().get_object(Bucket=DATA_BUCKET, Key=prediction)["Body"].read()
    with MemoryFile(body) as memory, memory.open() as raster:
        codes = raster.read(1)
        try:
            colours = raster.colormap(1)
        except ValueError as exc:
            raise ValueError("the raster has no colour table") from exc
        nodata = raster.nodata
        names = json.loads(raster.tags().get("classes", "{}"))
        crs, resolution = str(raster.crs), raster.res[0]

    table = np.zeros((256, 4), dtype=np.uint8)
    for code, colour in colours.items():
        table[code] = colour
    if nodata is not None:
        table[int(nodata)] = (0, 0, 0, 0)  # transparent

    counts = np.bincount(codes.ravel(), minlength=256)
    step = max(1, -(-max(codes.shape) // MAP_MAX_PIXELS))  # ceiling division
    return {
        "image": table[codes[::step, ::step]],
        "legend": [
            {
                "class_id": code,
                "name": names.get(str(code), f"Class {code}"),
                "colour": "#{:02x}{:02x}{:02x}".format(*table[code][:3]),
            }
            for code in np.flatnonzero(counts).tolist()
            if nodata is None or code != int(nodata)
        ],
        "caption": (
            f"{codes.shape[1]} × {codes.shape[0]} pixels at {resolution:g} m ({crs}), "
            f"shown at {resolution * step:g} m"
        ),
    }


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
        st.info("No predicted raster in the data bucket yet. Run `notebooks/05_scene_inference.ipynb`.")
    else:
        selected = st.selectbox("Prediction", keys)
        st.caption(f"s3://{DATA_BUCKET}/{selected}")
        try:
            land_cover = land_cover_map(selected)
        except Exception as exc:
            st.warning(f"Cannot render s3://{DATA_BUCKET}/{selected}: {exc}")
        else:
            image_column, legend_column = st.columns([4, 1])
            image_column.image(land_cover["image"], caption=land_cover["caption"], width="stretch")
            legend_column.markdown("**Land-cover class**")
            legend_column.markdown(
                "".join(
                    '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem">'
                    f'<span style="width:1rem;height:1rem;flex:none;border:1px solid #555;'
                    f'background:{entry["colour"]}"></span>'
                    f'<span>{entry["class_id"]}: {html.escape(entry["name"])}</span></div>'
                    for entry in land_cover["legend"]
                ),
                unsafe_allow_html=True,
            )

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
