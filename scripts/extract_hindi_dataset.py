import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASTORE = ROOT_DIR / "datastore"
TSV_PATH = DATASTORE / "cross_lingual" / "xnli" / "XNLI-1.0" / "xnli.dev.tsv"
OUT_DIR = DATASTORE / "cross_lingual" / "hindi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[1/4] Reading XNLI dataset from: {TSV_PATH}")
df = pd.read_csv(TSV_PATH, sep="\t", usecols=["language", "gold_label", "sentence1", "sentence2", "pairID"], low_memory=False)

# 1. Monolingual Hindi
print("[2/4] Extracting Hindi Monolingual pairs...")
df_hi = df[df["language"] == "hi"].dropna(subset=["sentence1", "sentence2", "gold_label"])
hi_mono_pairs = []
for _, row in df_hi.iterrows():
    label = row["gold_label"]
    if label in ["entailment", "contradiction", "neutral"]:
        score = 10.0 if label == "entailment" else (0.0 if label == "contradiction" else 4.0)
        hi_mono_pairs.append({
            "pairID": int(row["pairID"]),
            "original": str(row["sentence1"]).strip(),
            "simplification": str(row["sentence2"]).strip(),
            "score": score,
            "label": label,
            "language": "hi"
        })

mono_file = OUT_DIR / "hindi_monolingual.jsonl"
with open(mono_file, "w", encoding="utf-8") as f:
    for p in hi_mono_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"      -> Saved {len(hi_mono_pairs)} Hindi monolingual pairs to {mono_file.name}")

# Split 80/20 train/test
train_mono, test_mono = train_test_split(hi_mono_pairs, test_size=0.20, random_state=42)
with open(OUT_DIR / "train_hindi_mono.jsonl", "w", encoding="utf-8") as f:
    for p in train_mono:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(OUT_DIR / "test_hindi_mono.jsonl", "w", encoding="utf-8") as f:
    for p in test_mono:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"      -> Splits: Train={len(train_mono)} (80%), Test={len(test_mono)} (20%)")

# 2. Cross-Lingual English <-> Hindi
print("[3/4] Extracting Cross-Lingual English <-> Hindi pairs...")
df_en = df[df["language"] == "en"].set_index("pairID")
df_hi_idx = df_hi.set_index("pairID")
common_ids_en = df_en.index.intersection(df_hi_idx.index)

en_hi_pairs = []
for pid in common_ids_en:
    en_p = str(df_en.loc[pid, "sentence1"]).strip()
    hi_h = str(df_hi_idx.loc[pid, "sentence2"]).strip()
    label = df_en.loc[pid, "gold_label"]
    if label in ["entailment", "contradiction", "neutral"]:
        score = 10.0 if label == "entailment" else (0.0 if label == "contradiction" else 4.0)
        en_hi_pairs.append({
            "pairID": int(pid),
            "original": en_p,
            "simplification": hi_h,
            "score": score,
            "label": label,
            "src_lang": "en",
            "tgt_lang": "hi"
        })

en_hi_file = OUT_DIR / "crosslingual_en_hi.jsonl"
with open(en_hi_file, "w", encoding="utf-8") as f:
    for p in en_hi_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"      -> Saved {len(en_hi_pairs)} English <-> Hindi pairs to {en_hi_file.name}")

# 3. Cross-Lingual French <-> Hindi
print("[4/4] Extracting Cross-Lingual French <-> Hindi pairs...")
df_fr = df[df["language"] == "fr"].set_index("pairID")
common_ids_fr = df_fr.index.intersection(df_hi_idx.index)

fr_hi_pairs = []
for pid in common_ids_fr:
    fr_p = str(df_fr.loc[pid, "sentence1"]).strip()
    hi_h = str(df_hi_idx.loc[pid, "sentence2"]).strip()
    label = df_fr.loc[pid, "gold_label"]
    if label in ["entailment", "contradiction", "neutral"]:
        score = 10.0 if label == "entailment" else (0.0 if label == "contradiction" else 4.0)
        fr_hi_pairs.append({
            "pairID": int(pid),
            "original": fr_p,
            "simplification": hi_h,
            "score": score,
            "label": label,
            "src_lang": "fr",
            "tgt_lang": "hi"
        })

fr_hi_file = OUT_DIR / "crosslingual_fr_hi.jsonl"
with open(fr_hi_file, "w", encoding="utf-8") as f:
    for p in fr_hi_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"      -> Saved {len(fr_hi_pairs)} French <-> Hindi pairs to {fr_hi_file.name}")

print("\n✅ Successfully extracted and structured all Hindi datasets!")
