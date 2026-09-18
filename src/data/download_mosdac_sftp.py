"""
Automated MOSDAC SFTP Downloader & Ingestion Pipeline for CycloVision AI.

Connects to download.mosdac.gov.in via SFTP, downloads ordered INSAT-3DR
cyclone granules, and automatically processes them into the training dataset.

Usage:
    # Download and process the 20 most important peak lifecycle frames of Biparjoy:
    python -m src.data.download_mosdac_sftp --sample 20

    # Download all matched best-track lifecycle frames (~97 frames, ~1 GB):
    python -m src.data.download_mosdac_sftp --all-fixes
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import paramiko

from src.config import INSAT_DIR
from src.data.load_besttrack import load_observations
from src.data.process_insat_mosdac import process_and_save_frame, parse_timestamp_from_filename


MOSDAC_HOST = "download.mosdac.gov.in"
MOSDAC_PORT = 22
ORDER_SUBDIR = "Order/Sep26_190662"


def connect_sftp(username: str, password: str) -> tuple[paramiko.Transport, paramiko.SFTPClient]:
    """Connect to MOSDAC SFTP server."""
    transport = paramiko.Transport((MOSDAC_HOST, MOSDAC_PORT))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def download_and_ingest_files(
    username: str,
    password: str,
    storm_id: str = "2023-003",
    sample_count: int | None = 15,
):
    """
    Download INSAT-3DR H5 files from MOSDAC order and process them into .npy files.
    """
    raw_cache_dir = INSAT_DIR / "mosdac_raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {MOSDAC_HOST}:{MOSDAC_PORT} as {username}...")
    transport, sftp = connect_sftp(username, password)

    try:
        print(f"Listing files in remote order directory: {ORDER_SUBDIR}...")
        remote_files = sftp.listdir(ORDER_SUBDIR)
        print(f"Found {len(remote_files)} total granules in MOSDAC order.")

        # Filter only H5 files
        h5_files = [f for f in remote_files if f.endswith(".h5") or f.endswith(".nc")]
        h5_files.sort()

        if sample_count is not None and sample_count < len(h5_files):
            # Pick evenly spaced files across the entire lifecycle
            indices = np.linspace(0, len(h5_files) - 1, sample_count, dtype=int)
            selected_files = [h5_files[i] for i in indices]
        else:
            selected_files = h5_files

        print(f"\n[DOWNLOAD QUEUE] Downloading and processing {len(selected_files)} INSAT-3DR granules for storm {storm_id}...")

        success_count = 0
        for i, fname in enumerate(selected_files, 1):
            remote_path = f"{ORDER_SUBDIR}/{fname}"
            local_path = raw_cache_dir / fname

            print(f"[{i}/{len(selected_files)}] Downloading: {fname}...")
            if not local_path.exists():
                sftp.get(remote_path, str(local_path))

            # Process into .npy dataset
            try:
                out_npy = process_and_save_frame(local_path, storm_id)
                success_count += 1
            except Exception as e:
                print(f"[WARN] Failed processing {fname}: {e}")

        print("\n" + "=" * 60)
        print(f"MOSDAC Ingestion Complete: {success_count}/{len(selected_files)} real satellite frames processed into data/raw/insat/{storm_id}/")
        print("=" * 60)

    finally:
        sftp.close()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description="Automated MOSDAC SFTP Downloader")
    parser.add_argument("--user", type=str, default="jagomohan", help="MOSDAC username")
    parser.add_argument("--password", type=str, default="Jago@@31123900", help="MOSDAC password")
    parser.add_argument("--storm-id", type=str, default="2023-003", help="Target storm ID")
    parser.add_argument("--sample", type=int, default=15, help="Number of representative lifecycle frames to download")
    parser.add_argument("--all-fixes", action="store_true", help="Download all files in order")

    args = parser.parse_args()

    count = None if args.all_fixes else args.sample
    download_and_ingest_files(
        username=args.user,
        password=args.password,
        storm_id=args.storm_id,
        sample_count=count,
    )


if __name__ == "__main__":
    main()
