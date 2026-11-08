# Jupyter (phase 1) image: JupyterLab scipy stack plus the geospatial and MLOps
# libraries the notebooks need (rasterio, geopandas, mlflow, optuna, S3 and
# PostGIS clients).
#
# Build context is the repository root, consistent with the other services:
#   docker compose build jupyter
#   docker build -f apps/jupyter/Dockerfile -t sentinel2-jupyter .
FROM quay.io/jupyter/scipy-notebook:python-3.12

COPY --chown=${NB_UID}:${NB_GID} apps/jupyter/requirements.txt /tmp/requirements.txt

# pip into the conda environment of the image; fix-permissions keeps the
# environment writable by the notebook user (jovyan).
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Notebooks (./notebooks) and local data (./data) are bind-mounted here.
RUN mkdir -p /home/${NB_USER}/work /home/${NB_USER}/data
WORKDIR /home/${NB_USER}/work
