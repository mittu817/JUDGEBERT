# ⚖️🌐 JUDGEBERT: Universal & Legal Meaning Preservation Evaluator

An interactive neuro-symbolic platform for assessing semantic meaning preservation, factual faithfulness, and legal fidelity across **General Context (News, Science, Facts, Paraphrasing)** and **Legal & Contractual Clauses (FrJUDGE, ContractNLI, Insurance)** in **English**, **French**, and **Cross-Lingual (FR ↔ EN)** pairs.

Based on the research paper:  
**"JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences"**  
*Beauchemin, Albert-Rochette, Khoury, Déziel (EMNLP 2025)* — generalized with token-level Cross-Encoder NLI, Bidirectional Mutual Entailment ($A \iff B$), Syntactic Dependency Guardrails, and Closed-Loop Auto-Repair.

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

### 3. Run Automated Regression Test Suite
Verify all 17 end-to-end evaluation and closed-loop self-correction tests:

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

---

## 🏋️ Model Training & Fine-Tuning

To train or fine-tune the multilingual contrastive bi-encoder (`judgebert-multilingual-v2`) on FrJUDGE, Stanford ContractNLI, and Meta XNLI:

```bash
python finetune_judgebert_contrastive.py
```

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
│   ├── cross_lingual/        # Meta XNLI (FR ↔ EN multilingual pairs)
│   └── corpus_coling/        # French simplification parallel corpora
├── finetune_judgebert_contrastive.py  # Contrastive fine-tuning pipeline
├── train_and_evaluate_multilingual.py # Evaluation & benchmarking script
├── download_and_setup_datasets.py     # Automated dataset downloader
├── verify_demo_server.py     # 17-test end-to-end automated regression test suite
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
```
