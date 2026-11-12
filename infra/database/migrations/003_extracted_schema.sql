-- Training database · schema `extracted`.
--
-- Middle layer between the source files (schema `raw`) and the training
-- dataset (schema `training`): the pixels of the scene that fall inside the
-- ground-truth polygons, with nothing filtered out yet. Loaded by the Airflow
-- DAG extract_training_samples.

CREATE SCHEMA IF NOT EXISTS extracted;

-- One row per pixel whose centre lies inside a polygon of raw.polygon_classes.
-- `dataset` identifies the scene the bands belong to (reloading a dataset
-- replaces its rows) and `source` the polygons file. A pixel covered by two
-- polygons appears once per polygon. The bands hold the digital numbers of the
-- 20 m files as they come; `is_nodata` marks the pixels where a band has no
-- data. Cleaning (no-data pixels, pixels claimed by two classes, duplicates)
-- happens when training.samples is built from this table.
CREATE TABLE IF NOT EXISTS extracted.polygon_pixels (
    dataset      TEXT             NOT NULL,
    source       TEXT             NOT NULL,
    fid          INTEGER          NOT NULL,
    class_id     SMALLINT         NOT NULL,
    pixel_row    INTEGER          NOT NULL,
    pixel_col    INTEGER          NOT NULL,
    latitude     DOUBLE PRECISION NOT NULL,
    longitude    DOUBLE PRECISION NOT NULL,
    b01          INTEGER          NOT NULL,
    b02          INTEGER          NOT NULL,
    b03          INTEGER          NOT NULL,
    b04          INTEGER          NOT NULL,
    b05          INTEGER          NOT NULL,
    b06          INTEGER          NOT NULL,
    b07          INTEGER          NOT NULL,
    b8a          INTEGER          NOT NULL,
    b11          INTEGER          NOT NULL,
    b12          INTEGER          NOT NULL,
    is_nodata    BOOLEAN          NOT NULL,
    extracted_at TIMESTAMPTZ      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset, source, fid, pixel_row, pixel_col)
);

CREATE INDEX IF NOT EXISTS polygon_pixels_pixel_idx
    ON extracted.polygon_pixels (dataset, pixel_row, pixel_col);
