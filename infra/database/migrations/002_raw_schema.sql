-- Training database · schema `raw`.
--
-- Source files landed as they come, before any extraction or cleaning. Loaded
-- by the Airflow DAG load_polygons.

CREATE SCHEMA IF NOT EXISTS raw;

-- One row per ground-truth polygon of the TSV (columns WKT, fid, class).
-- `source` is the name of the file the row came from, so several versions of
-- the polygons can coexist; reloading a file replaces its rows. The geometry
-- keeps the CRS of the scene (tile T18NWL: WGS 84 / UTM zone 18N) and is not
-- repaired: raw.polygon_classes holds whatever the file says.
CREATE TABLE IF NOT EXISTS raw.polygon_classes (
    source    TEXT        NOT NULL,
    fid       INTEGER     NOT NULL,
    class_id  SMALLINT    NOT NULL,
    geom      GEOMETRY(Polygon, 32618) NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, fid)
);

CREATE INDEX IF NOT EXISTS polygon_classes_class_idx ON raw.polygon_classes (source, class_id);
CREATE INDEX IF NOT EXISTS polygon_classes_geom_idx  ON raw.polygon_classes USING GIST (geom);
