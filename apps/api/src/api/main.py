"""Optional inference API: classify one pixel from its ten band values.

Serves models:/<MODEL_NAME>@<MODEL_ALIAS> from the MLflow registry. Until a
model carries the alias, /predict/pixel answers 503; POST /model/reload retries
without restarting the container.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
BAND_NAMES = [band.strip() for band in os.environ["BAND_NAMES"].split(",")]

state: dict = {"model": None, "version": None, "error": None}


def load_model() -> None:
    try:
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        state["version"] = version.version
        state["error"] = None
        log.info("loaded %s (version %s)", MODEL_URI, version.version)
    except Exception as exc:  # registry empty, alias missing or MLflow down
        state["model"], state["version"], state["error"] = None, None, str(exc)
        log.warning("model unavailable: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Sentinel-2 land-cover inference API",
    description="Pixel-level land-cover classification with the production model.",
    lifespan=lifespan,
)


class PixelRequest(BaseModel):
    bands: dict[str, float] = Field(
        description=f"Reflectance of every band, keys: {', '.join(BAND_NAMES)}",
        examples=[{band: 0.1 for band in BAND_NAMES}],
    )


class PixelResponse(BaseModel):
    class_id: int
    model: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": state["version"],
        "status": "ready" if state["model"] is not None else "unavailable",
        "error": state["error"],
        "band_names": BAND_NAMES,
    }


@app.post("/model/reload")
def reload_model() -> dict:
    load_model()
    return model_info()


@app.post("/predict/pixel", response_model=PixelResponse)
def predict_pixel(request: PixelRequest) -> PixelResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=f"no model with alias {MODEL_ALIAS}: {state['error']}")
    missing = [band for band in BAND_NAMES if band not in request.bands]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing bands: {missing}")
    features = pd.DataFrame([[request.bands[band] for band in BAND_NAMES]], columns=BAND_NAMES)
    prediction = state["model"].predict(features)
    return PixelResponse(
        class_id=int(pd.Series(prediction).iloc[0]),
        model={"name": MODEL_NAME, "alias": MODEL_ALIAS, "version": state["version"]},
    )
