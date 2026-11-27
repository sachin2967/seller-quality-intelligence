"""Acquire the official public Olist v2 files without replacing existing imports."""

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce?datasetVersionNumber=2"
EXPECTED_ROWS = {
    "olist_orders_dataset.csv": 99441,
    "olist_order_items_dataset.csv": 112650,
    "olist_order_reviews_dataset.csv": 99224,
    "olist_sellers_dataset.csv": 3095,
}
EXPECTED_SHA256 = {
    "olist_orders_dataset.csv": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_order_items_dataset.csv": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_order_reviews_dataset.csv": "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
    "olist_sellers_dataset.csv": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
}


def main():
    destination = ROOT / "data/raw/official_v2"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "olist_v2.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with requests.get(SOURCE, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            total = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise ValueError(
                            "Dataset archive exceeded the 100 MiB download bound."
                        )
                    stream.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle returned a non-ZIP response.")
        temporary.replace(archive)
    observations = {}
    with zipfile.ZipFile(archive) as source:
        for name, expected in EXPECTED_ROWS.items():
            info = source.getinfo(name)  # Extract only exact, allowlisted member names.
            if info.file_size > 100 * 1024 * 1024:
                raise ValueError("Dataset member exceeded the 100 MiB bound.")
            path = destination / name
            if not path.exists():
                temporary = path.with_suffix(".csv.download")
                with source.open(info) as member, temporary.open("wb") as target:
                    shutil.copyfileobj(member, target)
                temporary.replace(path)
            rows = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=20000))
            if rows != expected:
                raise ValueError(
                    f"Unexpected row count in {name}: {rows}, expected {expected}"
                )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256[name]:
                raise ValueError(f"Olist v2 file checksum mismatch: {name}")
            observations[name] = {
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
    manifest = {
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "kaggle_ref": "olistbr/brazilian-ecommerce",
        "dataset_version": 2,
        "source_url": SOURCE,
        "source_page": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "license_observed": "CC BY-NC-SA 4.0",
        "license_metadata_source": "https://www.kaggle.com/api/v1/datasets/list?search=olistbr%2Fbrazilian-ecommerce",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": observations,
        "local_directory": "data/raw/official_v2",
        "integrity_scope": "Observed local hashes and known release row counts; no publisher-signed checksum was provided.",
        "redistribution": "Raw archive/CSVs remain ignored by Git; original subset imports are preserved. Respect the dataset's attribution, noncommercial and share-alike terms.",
    }
    output = ROOT / "docs/OLIST_SOURCE_MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
