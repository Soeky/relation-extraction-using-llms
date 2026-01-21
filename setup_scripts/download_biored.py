#!/usr/bin/env python
"""
Download BioRED dataset from NCBI FTP server.
"""

import urllib.request
import zipfile
from pathlib import Path


BIORED_ZIP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"


def download_file(url: str, dest_path: Path) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Done: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "BIORED.zip"

    print("Downloading BioRED dataset from NCBI FTP...")
    print(f"Target directory: {data_dir}")
    print()

    if not zip_path.exists():
        download_file(BIORED_ZIP_URL, zip_path)
    else:
        print(f"Skipping download: {zip_path.name} already exists")

    print()
    print("Extracting ZIP archive...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if member.endswith('.JSON') or member.endswith('.json'):
                filename = Path(member).name
                dest_path = data_dir / filename
                if not dest_path.exists():
                    print(f"  Extracting: {filename}")
                    with zf.open(member) as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())
                else:
                    print(f"  Skipping: {filename} (already exists)")

    print()
    print("Download and extraction complete!")
    print()
    print("Extracted files:")
    for f in sorted(data_dir.glob("*.JSON")) + sorted(data_dir.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
