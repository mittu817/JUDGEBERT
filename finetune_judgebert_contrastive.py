import os
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

# Offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent
DATASTORE = ROOT_DIR / "datastore"
OUTPUT_MODEL_DIR = ROOT_DIR / "demo_app" / "models" / "judgebert-multilingual-v2"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class LegalContrastiveDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {
            "source": item["source"],
            "target": item["target"],
            "score": torch.tensor(item["score"], dtype=torch.float32),
            "label_type": item.get("label_type", "regression"),
            "dataset": item.get("dataset", "unknown")
        }


def load_frjudge_data():
    """Loads French legal simplification pairs from FrJUDGE training files."""
    pairs = []
    seen = set()

    files = [
        DATASTORE / "training" / "train_FrJUDGE_with_data_augmentation.jsonl",
        DATASTORE / "training" / "train_FrJUDGE.jsonl",
        DATASTORE / "training" / "dev_FrJUDGE.jsonl"
    ]

    for fpath in files:
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    orig = data.get("original", "").strip()
                    simp = data.get("simplification", "").strip()
                    # Scale is 0 to 10 in FrJUDGE -> normalize to [0.0, 1.0]
                    raw_score = float(data.get("label", 5.0))
                    norm_score = max(0.0, min(1.0, raw_score / 10.0))

                    if orig and simp and (orig, simp) not in seen:
                        seen.add((orig, simp))
                        pairs.append({
                            "source": orig,
                            "target": simp,
                            "score": norm_score,
                            "label_type": "continuous",
                            "dataset": "FrJUDGE"
                        })
                except Exception:
                    continue

    print(f"[DATA] Loaded {len(pairs)} FrJUDGE French legal pairs.")
    return pairs


def load_contract_nli_data(max_samples=600):
    """Loads English contract entailment and contradiction pairs from ContractNLI."""
    pairs = []
    cnli_path = DATASTORE / "monolingual_english" / "contract_nli" / "contract-nli" / "train.json"
    if not cnli_path.exists():
        return pairs

    with open(cnli_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_info = data["labels"]
    seen = set()

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
                    if clause_text and len(clause_text) > 15:
                        if choice == "Entailment":
                            score = 1.0
                        elif choice == "Contradiction":
                            score = 0.0
                        else:
                            score = 0.35

                        key = (clause_text[:100], hyp)
                        if key not in seen:
                            seen.add(key)
                            pairs.append({
                                "source": clause_text,
                                "target": hyp,
                                "score": score,
                                "label_type": "entailment" if choice == "Entailment" else "contradiction",
                                "dataset": "ContractNLI"
                            })

    random.seed(42)
    random.shuffle(pairs)
    selected = pairs[:max_samples]
    print(f"[DATA] Loaded {len(selected)} ContractNLI English legal pairs.")
    return selected


def load_xnli_data(max_samples=400):
    """Loads cross-lingual French <-> English entailment & contradiction pairs from XNLI."""
    pairs = []
    tsv_path = DATASTORE / "cross_lingual" / "xnli" / "XNLI-1.0" / "xnli.dev.tsv"
    if not tsv_path.exists():
        return pairs

    df = pd.read_csv(tsv_path, sep="\t", usecols=["language", "gold_label", "sentence1", "sentence2", "pairID"], low_memory=False)
    df_en = df[df["language"] == "en"].set_index("pairID")
    df_fr = df[df["language"] == "fr"].set_index("pairID")
    common_ids = list(df_en.index.intersection(df_fr.index))

    for pid in common_ids:
        fr_p = str(df_fr.loc[pid, "sentence1"]).strip()
        en_h = str(df_en.loc[pid, "sentence2"]).strip()
        label = df_en.loc[pid, "gold_label"]
        if label in ["entailment", "contradiction"]:
            score = 1.0 if label == "entailment" else 0.0
            pairs.append({
                "source": fr_p,
                "target": en_h,
                "score": score,
                "label_type": label,
                "dataset": "XNLI"
            })

    random.seed(42)
    random.shuffle(pairs)
    selected = pairs[:max_samples]
    print(f"[DATA] Loaded {len(selected)} XNLI Cross-Lingual FR<->EN pairs.")
    return selected


def load_hindi_xnli_data(max_samples=400):
    """Loads Hindi cross-lingual and monolingual pairs from Meta XNLI."""
    pairs = []
    hi_train = DATASTORE / "cross_lingual" / "hindi" / "train_hindi_mono.jsonl"
    if hi_train.exists():
        with open(hi_train, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    orig = data.get("original", "").strip()
                    simp = data.get("simplification", "").strip()
                    lbl = data.get("label", "")
                    raw_score = float(data.get("score", 5.0))
                    if lbl in ["entailment", "contradiction"]:
                        score = 1.0 if lbl == "entailment" else 0.0
                        pairs.append({
                            "source": orig,
                            "target": simp,
                            "score": score,
                            "label_type": lbl,
                            "dataset": "Hindi_XNLI"
                        })
                except Exception:
                    continue

    random.seed(42)
    random.shuffle(pairs)
    selected = pairs[:max_samples]
    print(f"[DATA] Loaded {len(selected)} Hindi XNLI pairs.")
    return selected


def load_mildsum_data(max_samples=400):
    """Loads Indian court judgment simplification & contradiction pairs from MILDSum."""
    pairs = []
    mildsum_path = DATASTORE / "cross_lingual" / "mildsum" / "train_mildsum.jsonl"
    if not mildsum_path.exists():
        return pairs

    with open(mildsum_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                orig = data.get("original", "").strip()
                simp = data.get("simplification", "").strip()
                raw_score = float(data.get("score", 5.0))
                norm_score = max(0.0, min(1.0, raw_score / 10.0))
                lbl = data.get("label", "entailment")
                if orig and simp:
                    pairs.append({
                        "source": orig,
                        "target": simp,
                        "score": norm_score,
                        "label_type": lbl,
                        "dataset": "MILDSum"
                    })
            except Exception:
                continue

    random.seed(42)
    random.shuffle(pairs)
    selected = pairs[:max_samples]
    print(f"[DATA] Loaded {len(selected)} MILDSum Indian Legal judgment pairs.")
    return selected


def load_test_frjudge():
    """Loads held-out FrJUDGE test set for evaluation."""
    test_file = DATASTORE / "training" / "test_FrJUDGE.jsonl"
    pairs = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    pairs.append({
                        "source": data["original"].strip(),
                        "target": data["simplification"].strip(),
                        "score": float(data.get("label", 5.0)) / 10.0,
                        "raw_label": float(data.get("label", 5.0))
                    })
    print(f"[TEST] Loaded {len(pairs)} held-out FrJUDGE test pairs.")
    return pairs


def load_test_mildsum():
    """Loads held-out MILDSum Indian court judgment test set."""
    test_file = DATASTORE / "cross_lingual" / "mildsum" / "test_mildsum.jsonl"
    pairs = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    pairs.append({
                        "source": data["original"].strip(),
                        "target": data["simplification"].strip(),
                        "score": float(data.get("score", 5.0)) / 10.0,
                        "raw_label": float(data.get("score", 5.0))
                    })
    print(f"[TEST] Loaded {len(pairs)} held-out MILDSum Indian legal test pairs.")
    return pairs


class ContrastiveCosineLoss(nn.Module):
    """
    Combined Contrastive Margin + Cosine MSE Loss:
    - Minimizes MSE between cosine similarity and target score for continuous preservation.
    - Applies a contrastive margin penalty for hard contradictions (score <= 0.1).
    """
    def __init__(self, margin=0.25):
        super().__init__()
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, emb_src, emb_tgt, target_scores):
        cos_sim = nn.functional.cosine_similarity(emb_src, emb_tgt)
        # Direct regression loss
        loss_reg = self.mse(cos_sim, target_scores)

        # Contrastive margin penalty for contradictions
        is_neg = (target_scores <= 0.1).float()
        neg_margin_penalty = is_neg * torch.clamp(cos_sim - self.margin, min=0.0).pow(2)
        loss_contra = neg_margin_penalty.mean()

        return loss_reg + 1.5 * loss_contra


def evaluate_model_on_test(model, test_pairs):
    """Evaluates the model on held-out test data and returns Pearson r, Spearman rho, and RMSE."""
    model.eval()
    sources = [p["source"] for p in test_pairs]
    targets = [p["target"] for p in test_pairs]
    gold_scores = np.array([p["raw_label"] for p in test_pairs])

    with torch.no_grad():
        emb_src = model.encode(sources, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        emb_tgt = model.encode(targets, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    # Cosine similarity between normalized vectors
    cosines = np.sum(emb_src * emb_tgt, axis=1)
    # Scale from [-1, 1] / [0, 1] to [0, 10]
    pred_scores = np.clip(cosines * 10.0, 0.0, 10.0)

    pr, _ = pearsonr(gold_scores, pred_scores)
    sr, _ = spearmanr(gold_scores, pred_scores)
    rmse = np.sqrt(mean_squared_error(gold_scores, pred_scores))
    mae = mean_absolute_error(gold_scores, pred_scores)

    return {
        "pearson_r": round(float(pr), 3),
        "spearman_rho": round(float(sr), 3),
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3)
    }


def train_judgebert_contrastive(epochs=3, batch_size=8, lr=2e-5):
    print("=" * 70)
    print("  JUDGEBERT CONTRASTIVE FINE-TUNING PIPELINE (END-TO-END)")
    print("=" * 70)

    # 1. Gather all training datasets
    fr_pairs = load_frjudge_data()
    en_pairs = load_contract_nli_data(max_samples=500)
    xnli_pairs = load_xnli_data(max_samples=300)
    hi_pairs = load_hindi_xnli_data(max_samples=300)
    mildsum_pairs = load_mildsum_data(max_samples=350)

    all_train_pairs = fr_pairs + en_pairs + xnli_pairs + hi_pairs + mildsum_pairs
    random.seed(42)
    random.shuffle(all_train_pairs)
    print(f"\n[DATA] Total consolidated training pairs: {len(all_train_pairs)}")

    test_pairs = load_test_frjudge()
    mildsum_test_pairs = load_test_mildsum()

    # 2. Load base multilingual model
    base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\n[MODEL] Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name, local_files_only=True)

    # Baseline evaluation before fine-tuning
    print("\n--- BASELINE EVALUATION (Before Fine-Tuning) ---")
    base_metrics = evaluate_model_on_test(model, test_pairs)
    print(f"  FrJUDGE Baseline Pearson r:    {base_metrics['pearson_r']}")
    print(f"  FrJUDGE Baseline Spearman rho: {base_metrics['spearman_rho']}")
    print(f"  FrJUDGE Baseline RMSE (0-10):  {base_metrics['rmse']}")
    print(f"  FrJUDGE Baseline MAE:          {base_metrics['mae']}")

    if mildsum_test_pairs:
        base_mildsum = evaluate_model_on_test(model, mildsum_test_pairs)
        print(f"  MILDSum Baseline Pearson r:    {base_mildsum['pearson_r']}")
        print(f"  MILDSum Baseline Spearman rho: {base_mildsum['spearman_rho']}")
        print(f"  MILDSum Baseline RMSE (0-10):  {base_mildsum['rmse']}")

    # 3. Fine-tuning setup
    # Freeze lower token embeddings (layers 0-5) to preserve multilingual token foundations
    # Fine-tune upper contextual transformer layers (6-11) + pooling head
    transformer_module = model._first_module().auto_model
    if hasattr(transformer_module, "encoder") and hasattr(transformer_module.encoder, "layer"):
        layers = transformer_module.encoder.layer
        num_layers = len(layers)
        freeze_until = 6
        for i in range(freeze_until):
            for param in layers[i].parameters():
                param.requires_grad = False
        print(f"[OPT] Parameter-efficient fine-tuning: Frozen layers 0..{freeze_until-1}, fine-tuning layers {freeze_until}..{num_layers-1} + pooling head.")
    elif hasattr(transformer_module, "embeddings"):
        for param in transformer_module.embeddings.parameters():
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer with decoupled weight decay
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    loss_fn = ContrastiveCosineLoss(margin=0.20)

    # DataLoader
    dataset = LegalContrastiveDataset(all_train_pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n[TRAIN] Beginning fine-tuning on {device} ({epochs} epochs, batch size={batch_size}, {len(loader)} steps/epoch)...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, batch in enumerate(loader):
            src_texts = list(batch["source"])
            tgt_texts = list(batch["target"])
            target_scores = batch["score"].to(device)

            # Tokenize and encode through the model graph
            feat_src = model.tokenize(src_texts)
            feat_tgt = model.tokenize(tgt_texts)

            feat_src = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feat_src.items()}
            feat_tgt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feat_tgt.items()}

            out_src = model(feat_src)
            out_tgt = model(feat_tgt)

            emb_src = out_src["sentence_embedding"]
            emb_tgt = out_tgt["sentence_embedding"]

            # Compute contrastive loss
            loss = loss_fn(emb_src, emb_tgt, target_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

            if (step + 1) % 40 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - start_time
                avg_l = epoch_loss / step_count
                print(f"  Epoch [{epoch+1}/{epochs}] | Step [{step+1}/{len(loader)}] | Loss: {avg_l:.4f} | Elapsed: {elapsed:.1f}s")

        # Evaluate after each epoch
        test_metrics = evaluate_model_on_test(model, test_pairs)
        print(f"  >> Epoch {epoch+1} Test Eval: Pearson r={test_metrics['pearson_r']} (base: {base_metrics['pearson_r']}) | RMSE={test_metrics['rmse']} (base: {base_metrics['rmse']})")

    total_time = time.time() - start_time
    print(f"\n[TRAIN] Fine-tuning finished in {total_time:.1f} seconds.")

    # 4. Final Evaluation & Metric Gains
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON HELD-OUT FrJUDGE BENCHMARK")
    print("=" * 70)
    final_metrics = evaluate_model_on_test(model, test_pairs)
    delta_r = final_metrics["pearson_r"] - base_metrics["pearson_r"]
    delta_rmse = base_metrics["rmse"] - final_metrics["rmse"]

    print(f"  Final Pearson Correlation (r):   {final_metrics['pearson_r']} (Gain: +{delta_r:.3f})")
    print(f"  Final Spearman Rank (rho):       {final_metrics['spearman_rho']}")
    print(f"  Final Root Mean Squared Error:   {final_metrics['rmse']} (Reduction: -{delta_rmse:.3f})")
    print(f"  Final Mean Absolute Error:       {final_metrics['mae']}")

    mildsum_summary = {}
    if mildsum_test_pairs:
        print("\n" + "=" * 70)
        print("  FINAL EVALUATION ON HELD-OUT MILDSum (INDIAN LEGAL JUDGMENTS)")
        print("=" * 70)
        final_mildsum = evaluate_model_on_test(model, mildsum_test_pairs)
        print(f"  Final MILDSum Pearson r:         {final_mildsum['pearson_r']} (base: {base_mildsum['pearson_r']})")
        print(f"  Final MILDSum Spearman rho:     {final_mildsum['spearman_rho']}")
        print(f"  Final MILDSum RMSE:             {final_mildsum['rmse']} (base: {base_mildsum['rmse']})")
        mildsum_summary = {
            "mildsum_test_samples": len(mildsum_test_pairs),
            "mildsum_baseline_pearson_r": base_mildsum["pearson_r"],
            "mildsum_final_pearson_r": final_mildsum["pearson_r"],
            "mildsum_baseline_rmse": base_mildsum["rmse"],
            "mildsum_final_rmse": final_mildsum["rmse"]
        }

    # 5. Save the fine-tuned model checkpoint
    print(f"\n[SAVE] Saving fine-tuned JUDGEBERT model to: {OUTPUT_MODEL_DIR}")
    model.save(str(OUTPUT_MODEL_DIR))

    # Also save metadata/config report
    summary = {
        "model_name": "JUDGEBERT-Multilingual-v2-Contrastive",
        "base_backbone": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_samples": len(all_train_pairs),
        "test_samples": len(test_pairs),
        "baseline_pearson_r": base_metrics["pearson_r"],
        "final_pearson_r": final_metrics["pearson_r"],
        "pearson_r_gain": round(delta_r, 3),
        "baseline_rmse": base_metrics["rmse"],
        "final_rmse": final_metrics["rmse"],
        "rmse_reduction": round(delta_rmse, 3)
    }
    summary.update(mildsum_summary)

    with open(OUTPUT_MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Successfully saved model and metadata summary!")
    return summary


if __name__ == "__main__":
    train_judgebert_contrastive(epochs=2, batch_size=8, lr=2.5e-5)
