import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sentence_transformers import SentenceTransformer

# Offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent
DATASTORE = ROOT_DIR / "datastore"
OUTPUT_DIR = ROOT_DIR / "demo_app"


def extract_features(orig_list, simp_list, model):
    print("      Encoding sentence vectors...")
    e_orig = model.encode(orig_list, normalize_embeddings=True, show_progress_bar=False)
    e_simp = model.encode(simp_list, normalize_embeddings=True, show_progress_bar=False)
    
    cos_sim = np.sum(e_orig * e_simp, axis=1, keepdims=True)
    diff = np.abs(e_orig - e_simp)
    prod = e_orig * e_simp
    
    lex_feats = []
    for o, s in zip(orig_list, simp_list):
        w_orig = [w.strip(".,;:?!'\"()[]{}").lower() for w in o.split()]
        w_simp = [w.strip(".,;:?!'\"()[]{}").lower() for w in s.split()]
        s_orig = set([w for w in w_orig if w])
        s_simp = set([w for w in w_simp if w])
        jaccard = len(s_orig & s_simp) / max(len(s_orig | s_simp), 1)
        overlap = len(s_orig & s_simp) / max(len(s_orig), 1)
        reverse_overlap = len(s_orig & s_simp) / max(len(s_simp), 1)
        len_ratio = min(len(s), len(o)) / max(len(s), len(o), 1)
        word_ratio = min(len(w_simp), len(w_orig)) / max(len(w_simp), len(w_orig), 1)
        lex_feats.append([jaccard, overlap, reverse_overlap, len_ratio, word_ratio])
        
    return np.hstack([cos_sim, diff, prod, np.array(lex_feats)])


def run_monolingual_english_pipeline(model):
    print("\n" + "="*65)
    print(" [1/2] TRAINING & EVALUATION: MONOLINGUAL ENGLISH (CONTRACT-NLI)")
    print("="*65)
    
    cnli_path = DATASTORE / "monolingual_english" / "contract_nli" / "contract-nli" / "dev.json"
    with open(cnli_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    labels_info = data["labels"]
    pairs = []
    for doc in data["documents"]:
        text = doc["text"]
        spans = doc.get("spans", [])
        for ann_set in doc.get("annotation_sets", []):
            for label_key, ann in ann_set.get("annotations", {}).items():
                choice = ann.get("choice")
                span_indices = ann.get("spans", [])
                hyp = labels_info[label_key]["hypothesis"]
                if span_indices:
                    clause_parts = []
                    for idx in span_indices:
                        if idx < len(spans):
                            start, end = spans[idx]
                            clause_parts.append(text[start:end])
                    clause_text = " ".join(clause_parts).strip()
                    if clause_text and len(clause_text) > 10:
                        score = 10.0 if choice == "Entailment" else (0.0 if choice == "Contradiction" else 3.5)
                        binary = 1 if choice == "Entailment" else 0
                        pairs.append({
                            "original": clause_text,
                            "simplification": hyp,
                            "score": score,
                            "binary": binary,
                            "choice": choice
                        })
                        
    print(f"Total extracted legal pairs: {len(pairs)}")
    
    # Split 80% train / 20% test
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.20, random_state=42)
    print(f"Train size (80%): {len(train_pairs)} | Test size (20%): {len(test_pairs)}")
    
    orig_train = [p["original"] for p in train_pairs]
    simp_train = [p["simplification"] for p in train_pairs]
    y_train = np.array([p["score"] for p in train_pairs])
    y_train_bin = np.array([p["binary"] for p in train_pairs])
    
    orig_test = [p["original"] for p in test_pairs]
    simp_test = [p["simplification"] for p in test_pairs]
    y_test = np.array([p["score"] for p in test_pairs])
    y_test_bin = np.array([p["binary"] for p in test_pairs])
    
    X_train = extract_features(orig_train, simp_train, model)
    X_test = extract_features(orig_test, simp_test, model)
    
    # Train Ridge Regressor
    reg = Ridge(alpha=10.0)
    reg.fit(X_train, y_train)
    y_pred = np.clip(reg.predict(X_test), 0.0, 10.0)
    
    # Train Logistic Classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train_bin)
    y_pred_bin = clf.predict(X_test)
    
    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    pr, _ = pearsonr(y_test, y_pred)
    sr, _ = spearmanr(y_test, y_pred)
    acc = accuracy_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    
    print("\n--- Monolingual English Test Results (20% Split) ---")
    print(f"  RMSE (Error):                {rmse:.3f} (Scale: 0-10)")
    print(f"  MAE (Mean Absolute Error):    {mae:.3f}")
    print(f"  Pearson Correlation (r):      {pr:.3f}")
    print(f"  Spearman Correlation (rho):   {sr:.3f}")
    print(f"  Classification Accuracy:      {acc*100:.1f}%")
    print(f"  Classification F1-Score:      {f1*100:.1f}%")
    
    # Save model
    model_save_path = OUTPUT_DIR / "calibrator_english_contract.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump({"regressor": reg, "classifier": clf}, f)
    print(f"Saved English model to: {model_save_path}")
    
    return {
        "dataset": "ContractNLI (Monolingual English Legal)",
        "train_samples": len(train_pairs),
        "test_samples": len(test_pairs),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "pearson_r": round(pr, 3),
        "spearman_rho": round(sr, 3),
        "accuracy": round(acc * 100, 1),
        "f1_score": round(f1 * 100, 1)
    }


def run_cross_lingual_pipeline(model):
    print("\n" + "="*65)
    print(" [2/2] TRAINING & EVALUATION: CROSS-LINGUAL (FRENCH <-> ENGLISH XNLI)")
    print("="*65)
    
    tsv_path = DATASTORE / "cross_lingual" / "xnli" / "XNLI-1.0" / "xnli.dev.tsv"
    df = pd.read_csv(tsv_path, sep="\t", usecols=["language", "gold_label", "sentence1", "sentence2", "pairID"], low_memory=False)
    
    df_en = df[df["language"] == "en"].set_index("pairID")
    df_fr = df[df["language"] == "fr"].set_index("pairID")
    common_ids = df_en.index.intersection(df_fr.index)
    
    # Take French sentence1 as Premise, English sentence2 as Hypothesis
    pairs = []
    for pid in common_ids:
        fr_p = str(df_fr.loc[pid, "sentence1"])
        en_h = str(df_en.loc[pid, "sentence2"])
        label = df_en.loc[pid, "gold_label"]
        if label in ["entailment", "contradiction", "neutral"]:
            score = 10.0 if label == "entailment" else (0.0 if label == "contradiction" else 4.0)
            binary = 1 if label == "entailment" else 0
            pairs.append({
                "original": fr_p,
                "simplification": en_h,
                "score": score,
                "binary": binary,
                "label": label
            })
            
    print(f"Total cross-lingual French <-> English pairs: {len(pairs)}")
    
    # Split 80% train / 20% test
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.20, random_state=42)
    print(f"Train size (80%): {len(train_pairs)} | Test size (20%): {len(test_pairs)}")
    
    orig_train = [p["original"] for p in train_pairs]
    simp_train = [p["simplification"] for p in train_pairs]
    y_train = np.array([p["score"] for p in train_pairs])
    y_train_bin = np.array([p["binary"] for p in train_pairs])
    
    orig_test = [p["original"] for p in test_pairs]
    simp_test = [p["simplification"] for p in test_pairs]
    y_test = np.array([p["score"] for p in test_pairs])
    y_test_bin = np.array([p["binary"] for p in test_pairs])
    
    X_train = extract_features(orig_train, simp_train, model)
    X_test = extract_features(orig_test, simp_test, model)
    
    # Train Ridge Regressor
    reg = Ridge(alpha=10.0)
    reg.fit(X_train, y_train)
    y_pred = np.clip(reg.predict(X_test), 0.0, 10.0)
    
    # Train Logistic Classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train_bin)
    y_pred_bin = clf.predict(X_test)
    
    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    pr, _ = pearsonr(y_test, y_pred)
    sr, _ = spearmanr(y_test, y_pred)
    acc = accuracy_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    
    print("\n--- Cross-Lingual Test Results (20% Split) ---")
    print(f"  RMSE (Error):                {rmse:.3f} (Scale: 0-10)")
    print(f"  MAE (Mean Absolute Error):    {mae:.3f}")
    print(f"  Pearson Correlation (r):      {pr:.3f}")
    print(f"  Spearman Correlation (rho):   {sr:.3f}")
    print(f"  Classification Accuracy:      {acc*100:.1f}%")
    print(f"  Classification F1-Score:      {f1*100:.1f}%")
    
    # Save model
    model_save_path = OUTPUT_DIR / "calibrator_crosslingual_xnli.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump({"regressor": reg, "classifier": clf}, f)
    print(f"Saved Cross-Lingual model to: {model_save_path}")
    
    return {
        "dataset": "XNLI (French <-> English Cross-Lingual)",
        "train_samples": len(train_pairs),
        "test_samples": len(test_pairs),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "pearson_r": round(pr, 3),
        "spearman_rho": round(sr, 3),
        "accuracy": round(acc * 100, 1),
        "f1_score": round(f1 * 100, 1)
    }


if __name__ == "__main__":
    print("Loading Transformer Feature Extractor...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    
    res1 = run_monolingual_english_pipeline(model)
    res2 = run_cross_lingual_pipeline(model)
    
    print("\n" + "="*65)
    print(" SUMMARY OF 80/20 TRAINING & TESTING RESULTS")
    print("="*65)
    for r in [res1, res2]:
        print(f"Dataset: {r['dataset']}")
        print(f"  Train: {r['train_samples']} | Test: {r['test_samples']}")
        print(f"  Pearson r: {r['pearson_r']} | Spearman rho: {r['spearman_rho']}")
        print(f"  RMSE: {r['rmse']} | MAE: {r['mae']}")
        print(f"  Accuracy: {r['accuracy']}% | F1: {r['f1_score']}%")
        print("-" * 50)

    # FrJUDGE End-to-End Contrastive Fine-Tuning Summary
    summary_file = ROOT_DIR / "demo_app" / "models" / "judgebert-multilingual-v2" / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            cf_sum = json.load(f)
        print("Dataset: FrJUDGE (End-to-End Contrastive Fine-Tuning)")
        print(f"  Model: {cf_sum.get('model_name')}")
        print(f"  Train: {cf_sum.get('training_samples')} | Held-Out Test: {cf_sum.get('test_samples')}")
        print(f"  Baseline Pearson r: {cf_sum.get('baseline_pearson_r')} ➔ Fine-Tuned Pearson r: {cf_sum.get('final_pearson_r')} (+{cf_sum.get('pearson_r_gain')})")
        print(f"  Baseline RMSE: {cf_sum.get('baseline_rmse')} ➔ Fine-Tuned RMSE: {cf_sum.get('final_rmse')} (-{cf_sum.get('rmse_reduction')})")
        print("="*65)
