"""Training dataset construction (project brief, section 3.1): from the source
files to the table and the TSV the models are trained on.

    raw.polygon_classes + s3://<DATA_BUCKET>/bands/      source files, as loaded by
              |                                           load_polygons and upload_bands
              v  extract_polygon_pixels
    extracted.polygon_pixels                              every pixel inside a polygon,
              |                                           nothing filtered out
              v  build_training_samples
    training.samples                                      clean pixels, one row each
              |
              v  export_tsv
    s3://<DATA_BUCKET>/tsv/<dataset>.tsv                  Latitude, Longitude, bands, Class

A pixel belongs to a polygon when its centre lies inside it. The band values
are the digital numbers of the 20 m files, untouched. Every step replaces the
rows of its dataset in one transaction, so re-running is safe.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import re
import tempfile
from datetime import datetime

from airflow.sdk import Param, dag, task

log = logging.getLogger(__name__)

BANDS_PREFIX = "bands"
TSV_PREFIX = "tsv"
DEFAULT_SOURCE = "PolygonsClasses.tsv"
# <scene>_<band>_20m.jp2, e.g. T18NWL_20260810T152649_B8A_20m.jp2
BAND_FILE = re.compile(r"^(?P<scene>.+)_(?P<band>B[0-9A-Z]{2})_20m\.jp2$")
# Band columns of extracted.polygon_pixels and training.samples.
BAND_COLUMNS = ("b01", "b02", "b03", "b04", "b05", "b06", "b07", "b8a", "b11", "b12")
# Sentinel-2 marks missing data with 0 when the file does not declare a value.
DEFAULT_NODATA = 0


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


def _s3_client():
    """S3 client for MinIO; the credentials come from AWS_* in the environment."""
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])


def _band_names() -> list[str]:
    """Bands in the BAND_NAMES order, checked against the table columns."""
    names = [band.strip() for band in os.environ["BAND_NAMES"].split(",")]
    if sorted(name.lower() for name in names) != sorted(BAND_COLUMNS):
        raise ValueError(f"BAND_NAMES {names} does not match the table columns {BAND_COLUMNS}")
    return names


def _scene_bands(s3, bucket: str, scene: str, band_names: list[str]) -> tuple[str, dict[str, str]]:
    """Resolve the scene to extract and the object key of each of its bands.

    With no scene given, the bucket must hold the bands of exactly one."""
    scenes: dict[str, dict[str, str]] = {}
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{BANDS_PREFIX}/")
    for page in pages:
        for obj in page.get("Contents", []):
            match = BAND_FILE.match(obj["Key"].removeprefix(f"{BANDS_PREFIX}/"))
            if match:
                scenes.setdefault(match["scene"], {})[match["band"]] = obj["Key"]

    if not scene:
        if len(scenes) != 1:
            raise ValueError(
                f"expected the bands of one scene in s3://{bucket}/{BANDS_PREFIX}/, found "
                f"{sorted(scenes)}; run upload_bands or pass the `scene` param"
            )
        scene = next(iter(scenes))
    keys = scenes.get(scene, {})
    missing = [band for band in band_names if band not in keys]
    if missing:
        raise ValueError(f"scene {scene}: bands missing in s3://{bucket}/{BANDS_PREFIX}/: {missing}")
    return scene, {band: keys[band] for band in band_names}


def _polygon_pixels(polygons: list[tuple], transform, shape: tuple[int, int]):
    """Pixels whose centre lies inside each polygon.

    Returns four parallel arrays (fid, class_id, row, col). Each polygon is
    rasterised on its own small window, so a pixel covered by two polygons is
    reported once per polygon."""
    import numpy as np
    from affine import Affine
    from rasterio.features import geometry_mask

    height, width = shape
    to_pixel = ~transform
    fids, class_ids, rows, cols = [], [], [], []
    for fid, class_id, geom in polygons:
        min_x, min_y, max_x, max_y = geom.bounds
        col_a, row_a = to_pixel * (min_x, max_y)
        col_b, row_b = to_pixel * (max_x, min_y)
        col_start = max(math.floor(min(col_a, col_b)), 0)
        col_stop = min(math.ceil(max(col_a, col_b)), width)
        row_start = max(math.floor(min(row_a, row_b)), 0)
        row_stop = min(math.ceil(max(row_a, row_b)), height)
        if col_start >= col_stop or row_start >= row_stop:
            continue  # polygon outside the scene

        inside = geometry_mask(
            [geom],
            out_shape=(row_stop - row_start, col_stop - col_start),
            transform=transform * Affine.translation(col_start, row_start),
            invert=True,
        )
        window_rows, window_cols = np.nonzero(inside)
        fids.append(np.full(window_rows.size, fid))
        class_ids.append(np.full(window_rows.size, class_id))
        rows.append(window_rows + row_start)
        cols.append(window_cols + col_start)

    if not rows:
        return (np.empty(0, dtype=int),) * 4
    return tuple(np.concatenate(parts) for parts in (fids, class_ids, rows, cols))


@dag(
    dag_id="extract_training_samples",
    description="Extract the pixels inside the ground-truth polygons and build the training dataset and its TSV",
    schedule=None,  # manual trigger, after upload_bands and load_polygons
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "source": Param(
            DEFAULT_SOURCE,
            type="string",
            description="Polygons file to use, as stored in raw.polygon_classes.source",
        ),
        "scene": Param(
            "",
            type="string",
            description="Scene to extract (prefix of the band files); empty when the bucket holds one scene",
        ),
    },
    tags=["sentinel2", "ingestion"],
)
def extract_training_samples():
    @task
    def extract_polygon_pixels(params: dict | None = None) -> dict:
        """raw -> extracted: read the ten bands under every polygon."""
        import numpy as np
        import rasterio
        from rasterio.warp import transform as reproject_points
        from shapely import wkb

        source = params["source"]
        band_names = _band_names()
        bucket = os.environ["DATA_BUCKET"]
        s3 = _s3_client()
        dataset, band_keys = _scene_bands(s3, bucket, params["scene"], band_names)

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT fid, class_id, ST_AsBinary(geom), ST_SRID(geom) "
                "FROM raw.polygon_classes WHERE source = %s ORDER BY fid",
                (source,),
            )
            records = cur.fetchall()
        if not records:
            raise ValueError(f"no polygon of {source} in raw.polygon_classes; run load_polygons")
        polygons = [(fid, class_id, wkb.loads(bytes(geom))) for fid, class_id, geom, _ in records]
        polygons_srid = records[0][3]

        grid = None
        values: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory() as workdir:
            for band in band_names:
                path = os.path.join(workdir, f"{band}.jp2")
                s3.download_file(bucket, band_keys[band], path)
                with rasterio.open(path) as raster:
                    band_grid = (raster.crs, raster.transform, raster.shape)
                    if grid is None:
                        if raster.crs.to_epsg() != polygons_srid:
                            raise ValueError(
                                f"{band_keys[band]} is in {raster.crs}, the polygons in EPSG:{polygons_srid}"
                            )
                        grid = band_grid
                        fids, class_ids, rows, cols = _polygon_pixels(polygons, raster.transform, raster.shape)
                        if rows.size == 0:
                            raise ValueError(f"no pixel of {dataset} lies inside the polygons of {source}")
                        is_nodata = np.zeros(rows.size, dtype=bool)
                    elif band_grid != grid:
                        raise ValueError(f"{band_keys[band]} is not on the grid of the other bands")

                    values[band] = raster.read(1)[rows, cols]
                    nodata = DEFAULT_NODATA if raster.nodata is None else raster.nodata
                    is_nodata |= values[band] == nodata
                os.remove(path)
                log.info("sampled %s (%d pixels)", band_keys[band], rows.size)

        crs, transform, _ = grid
        xs, ys = transform * (cols + 0.5, rows + 0.5)  # pixel centres
        longitudes, latitudes = reproject_points(crs, "EPSG:4326", xs.tolist(), ys.tolist())

        buffer = io.StringIO()
        csv.writer(buffer, delimiter="\t", lineterminator="\n").writerows(
            zip(
                [dataset] * rows.size, [source] * rows.size,
                fids.tolist(), class_ids.tolist(), rows.tolist(), cols.tolist(),
                latitudes, longitudes,
                *(values[band].tolist() for band in band_names),
                is_nodata.tolist(),
            )
        )
        buffer.seek(0)
        columns = ", ".join(
            ["dataset", "source", "fid", "class_id", "pixel_row", "pixel_col", "latitude", "longitude"]
            + [band.lower() for band in band_names]
            + ["is_nodata"]
        )
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM extracted.polygon_pixels WHERE dataset = %s AND source = %s",
                (dataset, source),
            )
            cur.copy_expert(
                f"COPY extracted.polygon_pixels ({columns}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t')",
                buffer,
            )

        empty_polygons = len(polygons) - np.unique(fids).size
        log.info(
            "extracted %d pixels of %s from %d polygons of %s (%d polygons without a pixel, %d no-data pixels)",
            rows.size, dataset, len(polygons), source, empty_polygons, int(is_nodata.sum()),
        )
        return {"dataset": dataset, "source": source, "pixels": int(rows.size)}

    @task
    def build_training_samples(extracted: dict) -> dict:
        """extracted -> training: keep one row per clean pixel.

        Dropped: pixels with no data in a band and pixels claimed by polygons
        of different classes. A pixel covered twice by the same class is kept
        once."""
        dataset, source = extracted["dataset"], extracted["source"]
        bands = ", ".join(BAND_COLUMNS)
        band_values = ", ".join(f"min({band})" for band in BAND_COLUMNS)

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH pixels AS (
                    SELECT bool_or(is_nodata) AS is_nodata, count(DISTINCT class_id) AS classes
                    FROM extracted.polygon_pixels
                    WHERE dataset = %s AND source = %s
                    GROUP BY pixel_row, pixel_col
                )
                SELECT count(*),
                       count(*) FILTER (WHERE is_nodata),
                       count(*) FILTER (WHERE NOT is_nodata AND classes > 1)
                FROM pixels
                """,
                (dataset, source),
            )
            distinct_pixels, nodata_pixels, conflicting_pixels = cur.fetchone()

            cur.execute("DELETE FROM training.samples WHERE dataset = %s", (dataset,))
            # Every row of a pixel carries the same coordinates and band values,
            # so min() only collapses the duplicates.
            cur.execute(
                f"""
                INSERT INTO training.samples (dataset, latitude, longitude, {bands}, class_id)
                SELECT dataset, min(latitude), min(longitude), {band_values}, min(class_id)
                FROM extracted.polygon_pixels
                WHERE dataset = %s AND source = %s AND NOT is_nodata
                GROUP BY dataset, pixel_row, pixel_col
                HAVING count(DISTINCT class_id) = 1
                ORDER BY pixel_row, pixel_col
                """,
                (dataset, source),
            )
            samples = cur.rowcount
            cur.execute(
                "SELECT class_id, count(*) FROM training.samples WHERE dataset = %s "
                "GROUP BY class_id ORDER BY class_id",
                (dataset,),
            )
            per_class = dict(cur.fetchall())

        if samples == 0:
            raise ValueError(f"no clean pixel left for {dataset}")
        log.info(
            "training.samples: %d samples of %s from %d extracted rows (%d distinct pixels, "
            "%d dropped for no data, %d dropped for conflicting classes); samples per class: %s",
            samples, dataset, extracted["pixels"], distinct_pixels, nodata_pixels,
            conflicting_pixels, per_class,
        )
        return {"dataset": dataset, "samples": samples}

    @task
    def export_tsv(built: dict) -> str:
        """training -> TSV: Latitude, Longitude, the bands in the BAND_NAMES
        order and Class, uploaded to the data bucket."""
        dataset = built["dataset"]
        band_names = _band_names()
        columns = ", ".join(["latitude", "longitude", *(band.lower() for band in band_names), "class_id"])

        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {columns} FROM training.samples WHERE dataset = %s ORDER BY id", (dataset,))
            records = cur.fetchall()
        if len(records) != built["samples"]:
            raise ValueError(f"training.samples holds {len(records)} rows of {dataset}, expected {built['samples']}")

        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow(["Latitude", "Longitude", *band_names, "Class"])
        for latitude, longitude, *bands, class_id in records:
            # The digital numbers are integers stored in double columns.
            writer.writerow([latitude, longitude, *(int(v) if v.is_integer() else v for v in bands), class_id])

        bucket, key = os.environ["DATA_BUCKET"], f"{TSV_PREFIX}/{dataset}.tsv"
        body = buffer.getvalue().encode("utf-8")
        _s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/tab-separated-values")
        log.info("exported %d samples to s3://%s/%s (%d bytes)", len(records), bucket, key, len(body))
        return key

    export_tsv(build_training_samples(extract_polygon_pixels()))


extract_training_samples()
