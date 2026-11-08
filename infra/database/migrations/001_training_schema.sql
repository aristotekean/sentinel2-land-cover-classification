-- Training database · schema `training`.
--
-- The PostGIS extension is installed by infra/database/init on the first start
-- of the volume; this migration only creates the tables.

CREATE SCHEMA IF NOT EXISTS training;

-- One row per pixel extracted inside the ground-truth polygons (project brief,
-- section 3.1). Mirrors the TSV contract: Latitude, Longitude, the 10 spectral
-- bands (20 m, bottom-of-atmosphere reflectance) and the integer class label.
-- `dataset` identifies the TSV / extraction run the row came from, so several
-- versions of the dataset can coexist.
CREATE TABLE IF NOT EXISTS training.samples (
    id        BIGSERIAL PRIMARY KEY,
    dataset   TEXT             NOT NULL,
    latitude  DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    b01       DOUBLE PRECISION NOT NULL,
    b02       DOUBLE PRECISION NOT NULL,
    b03       DOUBLE PRECISION NOT NULL,
    b04       DOUBLE PRECISION NOT NULL,
    b05       DOUBLE PRECISION NOT NULL,
    b06       DOUBLE PRECISION NOT NULL,
    b07       DOUBLE PRECISION NOT NULL,
    b8a       DOUBLE PRECISION NOT NULL,
    b11       DOUBLE PRECISION NOT NULL,
    b12       DOUBLE PRECISION NOT NULL,
    class_id  SMALLINT         NOT NULL,
    geom      GEOMETRY(Point, 4326) GENERATED ALWAYS AS
                  (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED
);

CREATE INDEX IF NOT EXISTS samples_dataset_idx ON training.samples (dataset);
CREATE INDEX IF NOT EXISTS samples_class_idx   ON training.samples (dataset, class_id);
CREATE INDEX IF NOT EXISTS samples_geom_idx    ON training.samples USING GIST (geom);

-- Area of every class in a predicted raster (project brief, section 3.4),
-- computed after full-scene inference. `prediction` is the GeoTIFF key in the
-- MinIO data bucket (predictions/<file>.tif).
CREATE TABLE IF NOT EXISTS training.class_areas (
    prediction  TEXT             NOT NULL,
    class_id    SMALLINT         NOT NULL,
    pixel_count BIGINT           NOT NULL,
    area_ha     DOUBLE PRECISION NOT NULL,
    model_name  TEXT,
    model_version TEXT,
    computed_at TIMESTAMPTZ      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction, class_id)
);
