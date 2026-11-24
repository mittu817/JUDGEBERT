import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from demo_app.model_engine import JudgeBertEngine

engine = JudgeBertEngine()

pairs = [
    ("कंपनी ने इस तिमाही में भारी मुनाफ़े की सूचना दी।", "व्यापार ने पिछले तीन महीनों में बहुत सारा पैसा कमाया।", "Hindi Paraphrase"),
    ("The tenant must pay rent on the first day of each month.", "किराएदार को हर महीने की पहली तारीख को किराया देना होगा।", "Cross-Lingual EN <-> HI"),
    ("कर्मचारियों को ग्राहकों के पासवर्ड साझा करने की सख्त मनाही है।", "कर्मचारी आवश्यकता पड़ने पर ग्राहकों के पासवर्ड साझा कर सकते हैं।", "Hindi Contradiction"),
    ("भारत की राजधानी नई दिल्ली है।", "नई दिल्ली भारत की राजधानी है।", "Hindi Exact Equivalence")
]

for o, s, desc in pairs:
    res = engine.evaluate_pair(o, s)
    mode = res.get("mode")
    score = res.get("score")
    tier = res.get("tier")
    colbert = res["metrics"]["colbert_maxsim"]
    pooled = res["metrics"]["pooled_cosine"]
    verdict = res["nli"]["verdict"]
    
    print("=" * 65)
    print(f"[{desc}]")
    print(f"Orig: {o}")
    print(f"Simp: {s}")
    print(f"Mode: {mode}")
    print(f"ColBERT MaxSim: {colbert}% | Pooled Cosine: {pooled}%")
    print(f"Score: {score}/10.0 -> {tier}")
    print(f"NLI Verdict: {verdict}")
