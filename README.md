# Sentinel-2 land-cover classification · MLOps stack

Local MLOps platform for the ML101 final project: pixel-level land-cover classification of a Sentinel-2 L2A scene over Bogotá (tile T18NWL, 10 bands at 20 m). The stack runs with docker compose and mirrors the project phases: phase 1 (extraction, TSV dataset, training and HPO in Jupyter, tracked in MLflow) and phase 2 (Airflow DAGs to retrain and classify the full scene, a Streamlit dashboard and an optional FastAPI).

## Quick start

```bash
cp .env.example .env
docker compose up -d --build                                  # whole stack
docker compose up -d --build postgres minio mlflow jupyter    # phase 1 services only
docker compose down                                           # stop everything (-v also drops the data)
```

The first build downloads the geospatial stack (rasterio, geopandas) and takes a few minutes. On Linux set `AIRFLOW_UID=$(id -u)` in `.env` before starting Airflow.

## Architecture

```mermaid
flowchart TB
    minio[("MinIO<br/>bands · polygons · TSV · GeoTIFF · artifacts")]
    pg[("PostgreSQL + PostGIS<br/>mlflow · airflow · training")]
    jupyter["Jupyter<br/>extraction · TSV · training · HPO"]
    airflow["Airflow<br/>DAGs: retrain_model · predict_raster"]
    mlflow["MLflow<br/>tracking + model registry"]
    streamlit["Streamlit<br/>map · metrics · areas"]
    api["FastAPI (optional)<br/>pixel inference"]

    minio <--> jupyter
    pg <--> airflow
    pg <--> jupyter
    jupyter --> mlflow
    airflow --> mlflow
    airflow <--> minio
    mlflow <--> pg
    mlflow <--> minio
    mlflow --> streamlit
    mlflow --> api
```

| Service | Phase | URL | Credentials | Role |
|---------|-------|-----|-------------|------|
| `postgres` | 1 | `localhost:5432` | `postgres` / `postgres` | One PostGIS instance with the databases `mlflow`, `airflow` and `training` (own user each, see `.env.example`) |
| `minio` | 1 | console <http://localhost:9001>, S3 `localhost:9000` | `minioadmin` / `minioadmin` | Buckets `mlflow` (artifacts) and `sentinel2` (`bands/`, `polygons/`, `tsv/`, `predictions/`) |
| `mlflow` | 1 | <http://localhost:5001> | none | Tracking server and model registry; artifacts proxied to MinIO |
| `jupyter` | 1 | <http://localhost:8888/?token=sentinel2> | token `JUPYTER_TOKEN` | JupyterLab with rasterio, geopandas, scikit-learn, optuna, mlflow; `./notebooks` and `./data` mounted |
| `airflow-apiserver` | 2 | <http://localhost:8080> | `airflow` / `airflow` | DAGs `retrain_model` and `predict_raster` (skeletons in `apps/airflow/dags`) |
| `streamlit` | 2 | <http://localhost:8501> | none | Map, metrics of the production model and class areas |
| `api` | 2 (optional) | <http://localhost:8000/docs> | none | `POST /predict/pixel`, `GET /health`, `GET /model-info`, `POST /model/reload` |

Every value (ports, credentials, bucket and model names) lives in `.env`; compose has no defaults and stops on a missing variable.

## Data flow

1. **Extraction (Jupyter).** Read the ten 20 m bands and the training polygons (locally from `./data` or from `s3://sentinel2/bands/` and `polygons/`), sample the pixels inside each polygon and export the TSV with `Latitude`, `Longitude`, the bands in `BAND_NAMES` order and the class label. Upload it to `s3://sentinel2/tsv/` and, optionally, load it into `training.samples`.
2. **Training and HPO (Jupyter).** Train the six architectures, tune them and log params, metrics (overall accuracy, kappa, precision, recall, F1) and confusion matrices as MLflow runs of the experiment `MLFLOW_EXPERIMENT_NAME`. Register the winner as `MODEL_NAME` and give it the alias `MODEL_ALIAS` (`production`).
3. **Retrain and inference (Airflow).** `retrain_model` repeats step 2 from `training.samples`; `predict_raster` loads `models:/sentinel2-land-cover@production`, classifies the whole scene keeping the original CRS, uploads the GeoTIFF to `s3://sentinel2/predictions/` and stores the area per class in `training.class_areas`.
4. **Visualisation.** Streamlit shows the predicted raster, the metrics of the production model and the class areas; the API classifies a single pixel from its ten band values.

## Repository layout

```text
.
├── docker-compose.yml            # root stack; includes apps/airflow/compose.yml
├── .env.example                  # every variable of the stack
├── apps/
│   ├── mlflow/                   # official image + psycopg2 + boto3
│   ├── jupyter/                  # scipy-notebook + geospatial and MLOps libraries
│   ├── airflow/                  # image, compose slice and skeleton DAGs
│   ├── streamlit/                # dashboard
│   └── api/                      # optional FastAPI
├── infra/database/
│   ├── init/                     # creates the three databases and enables PostGIS
│   ├── migrate.sh                # applies migrations/ to the training database
│   └── migrations/               # training.samples, training.class_areas
├── notebooks/                    # phase 1 notebooks (mounted in Jupyter)
└── data/                         # local scene, polygons and rasters (git-ignored)
```

## Operation

| Task | How |
|------|-----|
| Open JupyterLab | <http://localhost:8888/?token=sentinel2>; notebooks persist in `./notebooks` |
| Copy the scene into MinIO | `mc` from the host or the MinIO console: bucket `sentinel2`, prefix `bands/` |
| Query the training database | `docker compose exec postgres psql -U training -d training` |
| Add a migration | New file in `infra/database/migrations` (name order) and `docker compose up train-db-migrate` |
| Trigger a DAG | Airflow UI, or `docker compose exec airflow-scheduler airflow dags trigger predict_raster` |
| Change the served model | Move the alias in the MLflow UI (Models → version → alias) or with `MlflowClient().set_registered_model_alias` |
| Reset everything | `docker compose down -v` (drops databases and buckets) |

## Troubleshooting

- `required variable ... is missing a value`: `.env` is missing or incomplete; compare it with `.env.example`.
- MLflow answers `403` from another machine: add the host to `MLFLOW_ALLOWED_HOSTS` and the origin to `MLFLOW_CORS_ALLOWED_ORIGINS`.
- `/predict/pixel` answers `503`: no model carries the `production` alias yet, or MLflow is unreachable; `GET /model-info` shows the error.
- Airflow cannot write `apps/airflow/logs` on Linux: set `AIRFLOW_UID` to `id -u` and recreate the containers.
- Database credentials changed in `.env` but login fails: they are applied when the volume is first created; run `down -v` or change them inside Postgres.
