import os
import sys
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASTORE = ROOT_DIR / "datastore"

CONTRACT_NLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
PAWS_X_URL = "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz"


def download_file(url: str, dest_path: Path):
    print(f"\n[1/2] Downloading from: {url}")
    print(f"      Destination: {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = round(downloaded / total_size * 100, 1)
                print(f"\r      Progress: {percent}% ({round(downloaded/1024/1024, 1)} MB)", end="")
    print(f"\n      Download completed successfully.")


def setup_contract_nli():
    dest_dir = DATASTORE / "monolingual_english" / "contract_nli"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "contract-nli.zip"

    if not any(dest_dir.glob("*.json")):
        download_file(CONTRACT_NLI_URL, zip_path)
        print(f"[2/2] Extracting ContractNLI to {dest_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        if zip_path.exists():
            zip_path.unlink()
        print(f"ContractNLI setup complete in: {dest_dir}")
    else:
        print(f"ContractNLI already extracted in {dest_dir}")


def setup_paws_x():
    dest_dir = DATASTORE / "cross_lingual" / "paws_x"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "x-final.tar.gz"

    if not any(dest_dir.glob("*")):
        download_file(PAWS_X_URL, tar_path)
        print(f"[2/2] Extracting PAWS-X to {dest_dir} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest_dir)
        if tar_path.exists():
            tar_path.unlink()
        print(f"PAWS-X setup complete in: {dest_dir}")
    else:
        print(f"PAWS-X already extracted in {dest_dir}")


if __name__ == "__main__":
    print("=" * 65)
    print(" 📥 DOWNLOADING MONOLINGUAL ENGLISH & CROSS-LINGUAL DATASETS")
    print("=" * 65)
    try:
        setup_contract_nli()
        setup_paws_x()
        print("\n" + "=" * 65)
        print(" ALL DATASETS DOWNLOADED AND EXTRACTED SUCCESSFULLY!")
        print(f" Datastore Location: {DATASTORE}")
        print("=" * 65)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)
