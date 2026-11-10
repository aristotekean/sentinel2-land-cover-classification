"""Band ingestion: upload the Sentinel-2 bands of the local scene archive to
the MinIO data bucket (s3://<DATA_BUCKET>/bands/).

The archive is the zip under ./data on the host, mounted read-only at
/opt/airflow/data. Each band is streamed out of the zip straight into MinIO,
so nothing is extracted to disk. Re-running is safe: a band already stored
with the same size is skipped.
"""

from __future__ import annotations

import logging
import os
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

from airflow.sdk import Param, dag, task

log = logging.getLogger(__name__)

DEFAULT_ARCHIVE = "/opt/airflow/data/imagery-sentinel2-bog-band.zip"
BANDS_PREFIX = "bands"
BAND_SUFFIX = ".jp2"


def _s3_client():
    """S3 client for MinIO; the credentials come from AWS_* in the environment."""
    import boto3

    return boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])


def _band_key(member: str) -> str:
    """Object key of a zip member: its file name under bands/, whatever folder
    the archive nests it in."""
    return f"{BANDS_PREFIX}/{PurePosixPath(member).name}"


@dag(
    dag_id="upload_bands",
    description="Upload the Sentinel-2 bands of the local zip archive to MinIO",
    schedule=None,  # manual trigger, once per scene
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "archive": Param(
            DEFAULT_ARCHIVE,
            type="string",
            description="Path of the zip archive inside the Airflow containers",
        ),
    },
    tags=["sentinel2", "ingestion"],
)
def upload_bands():
    @task
    def list_bands(params: dict | None = None) -> list[dict]:
        """List the bands of the archive, failing early if it is missing,
        corrupt or holds no band."""
        archive_path = params["archive"]
        with zipfile.ZipFile(archive_path) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"corrupt member in {archive_path}: {corrupt}")
            bands = [
                {"archive": archive_path, "member": info.filename, "size": info.file_size}
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(BAND_SUFFIX)
            ]
        if not bands:
            raise ValueError(f"no {BAND_SUFFIX} band found in {archive_path}")

        keys = [_band_key(band["member"]) for band in bands]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate band file names in {archive_path}: {sorted(keys)}")
        log.info("found %d bands in %s", len(bands), archive_path)
        return bands

    @task(retries=2)
    def upload_band(band: dict) -> dict:
        """Stream one band from the zip into the data bucket."""
        from botocore.exceptions import ClientError

        bucket, key = os.environ["DATA_BUCKET"], _band_key(band["member"])
        s3 = _s3_client()

        try:
            stored_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            stored_size = None

        if stored_size == band["size"]:
            log.info("skip s3://%s/%s: already stored (%d bytes)", bucket, key, stored_size)
        else:
            with zipfile.ZipFile(band["archive"]) as archive, archive.open(band["member"]) as stream:
                s3.upload_fileobj(stream, bucket, key)
            log.info("uploaded s3://%s/%s (%d bytes)", bucket, key, band["size"])
        return {"key": key, "size": band["size"]}

    @task
    def verify_upload(uploaded: list[dict]) -> None:
        """Check every band is in the bucket with the size it has in the zip."""
        bucket = os.environ["DATA_BUCKET"]
        pages = _s3_client().get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=f"{BANDS_PREFIX}/"
        )
        stored = {obj["Key"]: obj["Size"] for page in pages for obj in page.get("Contents", [])}

        mismatched = [band["key"] for band in uploaded if stored.get(band["key"]) != band["size"]]
        if mismatched:
            raise ValueError(f"bands missing or truncated in s3://{bucket}: {mismatched}")
        log.info("verified %d bands in s3://%s/%s/", len(uploaded), bucket, BANDS_PREFIX)

    verify_upload(upload_band.expand(band=list_bands()))


upload_bands()
