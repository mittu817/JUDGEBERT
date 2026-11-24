# ⚖️🌐 JUDGEBERT: Universal & Legal Meaning Preservation Evaluator

An interactive neuro-symbolic platform for assessing semantic meaning preservation, factual faithfulness, and legal fidelity across **General Context (News, Science, Facts, Paraphrasing)** and **Legal & Contractual Clauses (FrJUDGE, ContractNLI, Insurance)** in **English**, **French**, **Hindi**, and **Cross-Lingual (EN ↔ FR, EN ↔ HI, FR ↔ HI)** pairs.

Based on the research paper:  
**"JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences"**  
*Beauchemin, Albert-Rochette, Khoury, Déziel (EMNLP 2025)* — generalized with token-level Cross-Encoder NLI, Bidirectional Mutual Entailment ($A \iff B$), Syntactic Dependency Guardrails, Multilingual Hindi Support, and Closed-Loop Auto-Repair.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/mittu817/JUDGEBERT.git
cd JUDGEBERT
pip install -r requirements.txt
```

### 2. Run the Interactive Web App
Start the interactive evaluator:

```bash
python run_demo.py
```

Then open your browser at **[http://localhost:5000](http://localhost:5000)**.

> [!NOTE]
> **Model Weights (`*.safetensors`):**  
> By default, `.gitignore` excludes `*.safetensors` so the repository pushes smoothly without hitting GitHub's 100 MB single-file limit. Anyone who clones the repo can generate the fine-tuned weights in ~2 minutes with:
> ```bash
> python finetune_judgebert_contrastive.py
> ```
> If local fine-tuned weights are not generated, the engine will automatically rely on the base model fallback (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).

### 3. Run Automated Regression Test Suite
Verify all 21 end-to-end evaluation, Hindi cross-lingual, MILDSum Indian court judgment, and closed-loop self-correction tests:

```bash
python verify_demo_server.py
```

---

## 🎯 Architecture & Core Capabilities

### 🌍 1. General Context Meaning Preservation
- **Entity & Actor Inversion Defense:** Detects swapped entities or agent-patient reversals (e.g. *"Apple acquired NeXT"* vs *"NeXT acquired Apple"*).
- **Causal Directionality:** Verifies cause-and-effect flow (`because`, `therefore`, `parce que`) is not inverted (e.g. *"Match cancelled because thunderstorm"* vs *"Thunderstorm hit because match cancelled"*).
- **Factual Precision:** Protects against hallucinated dates, altered quantities (e.g. *"1889 / 330m"* vs *"1989 / 130m"*), and numbers across decimal notations (`8.9` vs `8,9`).
- **Science, Culture & News Paraphrasing:** Evaluates high-fidelity plain language rewrites (e.g. NASA James Webb Telescope, Louvre museum visitors).

### ⚖️ 2. Legal & Contractual Meaning Preservation
- **Actor-Role Binding & Party Inversion:** Catches contract obligation transfers (`Disclosing Party` ↔ `Receiving Party`, `Landlord` ↔ `Tenant`, `Assureur` ↔ `Assuré`).
- **Syntactic Dependency Guardrails:** Analyzes active vs passive subject-agent-patient dependencies to prevent false alarms on legitimate grammatical restructurings while strictly enforcing role fidelity.
- **Deontic Modals & Polarity Inversion:** Detects flipped obligations (`shall` vs `shall not` / `not required`, `doit` vs `ne doit pas` / `dispensé`).
- **Critical Exception Loss Detector:** Flags dropped carve-outs (`unless`, `except`, `provided that`, `sauf si`).
- **Legal Deadlines & Monetary Caps:** Strict verification of contract grace periods, penalties, and interest rates.

### 🔄 3. Bidirectional Mutual Entailment ($A \iff B$) & ColBERT Late Interaction
- **Forward ($A \implies B$):** Factual Faithfulness (verifies the rewrite introduces no hallucinations or contradictions).
- **Reverse ($B \implies A$):** Information Sufficiency (verifies the rewrite retains sufficient substance to entail the original).
- **ColBERT-Style Multi-Vector MaxSim:** Token-to-token contextual late interaction computing bidirectional token coverage ($A \to B$ coverage vs $B \to A$ sufficiency check) blended with dense sentence embeddings.
- **Calibrated Scoring Engine:** Balanced continuous transfer function ensuring natural plain-language paraphrases and synonymous rewrites are scored accurately without piecewise cliffs.

### 🛠️ 4. Closed-Loop Auto-Repair (Self-Correction Arbiter)
- **Automatic Diagnosis:** Pinpoints the exact root causes of meaning degradation (entity reversals, party swaps, constraint/date alterations, flipped deontic modals, omitted exceptions).
- **Surgical Patching & Canonical Synthesis:** Generates targeted corrections restoring factual grounding and legal validity.
- **Closed-Loop Verification:** Loops candidates through the Arbiter until achieving a score $\ge 9.0/10$ with zero guardrail violations.
- **One-Click Application:** Users can view the exact before-and-after score delta (e.g. $0.5 \rightarrow 10.0/10$) and apply the repaired sentence directly to the target.

### 🇮🇳 5. MILDSum Indian Legal Judgments & Multilingual Hindi Support
- **MILDSum Multilingual Indian Legal Judgments:** Integrated 424 benchmark pairs derived from authentic High Court and Supreme Court of India judgments (EMNLP 2023 `Law-AI/MILDSum`), covering NDPS Act Section 37 bail conditions, administrative writ petitions, matrimonial law, and retrospective orders.
- **Cross-Lingual Evaluation (EN ↔ HI):** Evaluates original English court judgments directly against professional Hindi legal practitioner summaries.
- **Monolingual Hindi Simplification (HI ↔ HI):** Scores Hindi legal formulations against plain-language Devanagari rewrites.
- **Indian Legal Deontic Guardrails:** Custom neuro-symbolic checks for Indian judicial rulings (bail granted vs dismissed, payment directives vs claim dismissals, retrospective bars).

---

## 🏋️ Model Training & Fine-Tuning

To train or fine-tune the multilingual contrastive bi-encoder (`judgebert-multilingual-v2`) on FrJUDGE, Stanford ContractNLI, Meta XNLI, and MILDSum:

```bash
python finetune_judgebert_contrastive.py
```
*This fine-tunes the contrastive objective across all French, English, and Hindi legal pairs and outputs weights in ~2 minutes.*

To run offline multilingual evaluation across all metrics (Pearson $r$, Spearman $\rho$, MSE, MAE, Classification Accuracy & F1):

```bash
python train_and_evaluate_multilingual.py
```

---

## 📁 Repository Structure

```
JUDGEBERT/
├── demo_app/                 # Interactive Web UI & Scoring Engine
│   ├── app.py                # Flask backend server & REST API (/api/evaluate, /api/repair)
│   ├── model_engine.py       # Multi-model inference engine (Bi-Encoder + Cross-Encoder NLI)
│   ├── legal_guardrails.py   # Universal & Legal neuro-symbolic guardrail engine
│   ├── auto_repair.py        # Closed-loop self-correction arbiter pipeline
│   ├── models/               # Model weights & configurations
│   └── templates/
│       └── index.html        # Interactive responsive frontend with domain selector & live cards
├── datastore/                # Benchmark Datasets
│   ├── released/
│   │   └── FrJUDGE.jsonl     # Full FrJUDGE dataset (1,485 expert annotations)
│   ├── monolingual_english/  # Stanford ContractNLI (English Legal)
│   ├── cross_lingual/
│   │   ├── xnli/             # Meta XNLI 1.0 (FR ↔ EN multilingual pairs)
│   │   ├── hindi/            # Hindi monolingual & Cross-Lingual (EN ↔ HI, FR ↔ HI) pairs
│   │   └── mildsum/          # MILDSum Indian Legal Judgments (EN ↔ HI, HI ↔ HI, Contradictions)
│   └── corpus_coling/        # French simplification parallel corpora
├── scripts/                  # Data extraction & evaluation utilities
│   ├── extract_hindi_dataset.py  # Hindi dataset extraction from XNLI
│   ├── download_mildsum.py       # MILDSum sample downloader
│   ├── process_mildsum.py        # MILDSum Indian court judgment parser & aligner
│   ├── test_hindi_engine.py      # Hindi Devanagari validation script
│   └── test_mildsum_engine.py    # MILDSum engine validation script
├── finetune_judgebert_contrastive.py  # Contrastive fine-tuning pipeline
├── train_and_evaluate_multilingual.py # Evaluation & benchmarking script
├── download_and_setup_datasets.py     # Automated dataset downloader
├── verify_demo_server.py     # 21-test end-to-end automated regression test suite
├── run_demo.py               # One-click demo server launcher
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore patterns
└── README.md
```

---

## 📜 Citation

```bibtex
@inproceedings{beauchemin2025judgebert,
  title={JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences},
  author={Beauchemin, David and Albert-Rochette, Michelle and Khoury, Richard and D{\'e}ziel, Pierre-Luc},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={92--118},
  year={2025}
}

@inproceedings{datta-etal-2023-mildsum,
  title={{MILDS}um: A Novel Benchmark Dataset for Multilingual Summarization of {I}ndian Legal Case Judgments},
  author={Datta, Debtanu and Soni, Shubham and Mukherjee, Rajdeep and Ghosh, Saptarshi},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={5291--5302},
  year={2023}
}
```
