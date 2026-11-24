import urllib.request
import os
from pathlib import Path

out_dir = Path("datastore/cross_lingual/mildsum")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Fetch README.md from MILDSum repo
readme_url = "https://raw.githubusercontent.com/Law-AI/MILDSum/main/README.md"
try:
    print(f"Fetching {readme_url}...")
    req = urllib.request.Request(readme_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        readme_content = resp.read().decode("utf-8")
        (out_dir / "README.md").write_text(readme_content, encoding="utf-8")
        print("README.md saved successfully.")
        print("--- README Preview ---")
        print("\n".join(readme_content.splitlines()[:40]))
        print("----------------------")
except Exception as e:
    print(f"Error fetching README: {e}")

# 2. Fetch Samples 1 to 10
languages = ["EN_Judgment.txt", "EN_Summary.txt", "HI_Summary.txt", "BN_Summary.txt", "GU_Summary.txt", "MR_Summary.txt", "TA_Summary.txt", "TE_Summary.txt"]
raw_base = "https://raw.githubusercontent.com/Law-AI/MILDSum/main/Data/MILDSum_Samples"

for i in range(1, 11):
    sample_name = f"Sample_{i}"
    sample_dir = out_dir / "samples" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in ["EN_Judgment.txt", "EN_Summary.txt", "HI_Summary.txt"]:
        file_url = f"{raw_base}/{sample_name}/{filename}"
        target_path = sample_dir / filename
        if not target_path.exists():
            try:
                req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    content = resp.read()
                    target_path.write_bytes(content)
                    print(f"Saved {sample_name}/{filename} ({len(content)} bytes)")
            except Exception as e:
                print(f"Failed to fetch {file_url}: {e}")
        else:
            print(f"Already exists: {sample_name}/{filename}")

print("Download complete.")
