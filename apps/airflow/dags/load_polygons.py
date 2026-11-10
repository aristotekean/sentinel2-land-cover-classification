"""Polygon ingestion: load the ground-truth polygons of the local TSV into the
training database, table raw.polygon_classes.

The TSV (columns WKT, fid, class) lives under ./data on the host, mounted
read-only at /opt/airflow/data. The file is validated before the database is
touched and loaded in one transaction that replaces the rows of the same
source file, so re-running is safe and a failed run leaves the table as it was.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import PurePosixPath

from airflow.sdk import Param, dag, task

log = logging.getLogger(__name__)

DEFAULT_POLYGONS = "/opt/airflow/data/PolygonsClasses.tsv"
EXPECTED_COLUMNS = ["WKT", "fid", "class"]
# CRS of the scene (tile T18NWL): WGS 84 / UTM zone 18N. The TSV does not
# declare it; it must match the SRID of raw.polygon_classes.geom.
POLYGONS_SRID = 32618


def _connect():
    """Connection to the training database (PostGIS)."""
    import psycopg2

    return psycopg2.connect(
        host=os.environ["TRAIN_DB_HOST"],
        port=os.environ["TRAIN_DB_PORT"],
        dbname=os.environ["TRAIN_DB_NAME"],
        user=os.environ["TRAIN_DB_USER"],
        password=os.environ["TRAIN_DB_PASSWORD"],
    )


def _read_polygons(path: str) -> list[tuple[int, int, str]]:
    """Parse the TSV into (fid, class_id, wkt) rows, failing on a wrong header,
    a non-integer fid or class, an empty geometry or a repeated fid."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"{path}: expected columns {EXPECTED_COLUMNS}, got {reader.fieldnames}")
        rows = []
        for line, record in enumerate(reader, start=2):
            try:
                fid, class_id = int(record["fid"]), int(record["class"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: fid and class must be integers") from error
            if not record["WKT"]:
                raise ValueError(f"{path}:{line}: empty WKT")
            rows.append((fid, class_id, record["WKT"]))

    if not rows:
        raise ValueError(f"{path}: no polygon found")
    if len({fid for fid, _, _ in rows}) != len(rows):
        raise ValueError(f"{path}: repeated fid")
    return rows


@dag(
    dag_id="load_polygons",
    description="Load the ground-truth polygons of the local TSV into raw.polygon_classes",
    schedule=None,  # manual trigger, once per version of the polygons
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "polygons": Param(
            DEFAULT_POLYGONS,
            type="string",
            description="Path of the polygons TSV inside the Airflow containers",
        ),
    },
    tags=["sentinel2", "ingestion"],
)
def load_polygons():
    @task
    def load_raw(params: dict | None = None) -> dict:
        """Validate the TSV and replace the rows of this source file."""
        from psycopg2.extras import execute_values

        path = params["polygons"]
        source = PurePosixPath(path).name
        rows = _read_polygons(path)

        with _connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM raw.polygon_classes WHERE source = %s", (source,))
            replaced = cur.rowcount
            execute_values(
                cur,
                "INSERT INTO raw.polygon_classes (source, fid, class_id, geom) VALUES %s",
                [(source, fid, class_id, wkt) for fid, class_id, wkt in rows],
                template=f"(%s, %s, %s, ST_GeomFromText(%s, {POLYGONS_SRID}))",
                page_size=1000,
            )
        log.info(
            "loaded %d polygons from %s into raw.polygon_classes (replaced %d)",
            len(rows), source, replaced,
        )
        return {"source": source, "rows": len(rows)}

    @task
    def verify_load(loaded: dict) -> None:
        """Check the table holds every polygon of the file and report the
        class distribution and the invalid geometries (kept as they come)."""
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT class_id, count(*), count(*) FILTER (WHERE NOT ST_IsValid(geom))
                FROM raw.polygon_classes
                WHERE source = %s
                GROUP BY class_id
                ORDER BY class_id
                """,
                (loaded["source"],),
            )
            per_class = cur.fetchall()

        stored = sum(count for _, count, _ in per_class)
        if stored != loaded["rows"]:
            raise ValueError(
                f"raw.polygon_classes holds {stored} rows of {loaded['source']}, expected {loaded['rows']}"
            )
        invalid = sum(count for _, _, count in per_class)
        if invalid:
            log.warning("%d invalid geometries in %s", invalid, loaded["source"])
        log.info(
            "verified %d polygons of %s; polygons per class: %s",
            stored, loaded["source"], {class_id: count for class_id, count, _ in per_class},
        )

    verify_load(load_raw())


load_polygons()
