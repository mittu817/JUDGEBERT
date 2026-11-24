import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_app.model_engine import JudgeBertEngine
from demo_app.legal_guardrails import LegalGuardrails
import pprint

engine = JudgeBertEngine()

orig = "The Receiving Parties undertake and agree: f) Not to alter, modify, disassemble, reverse engineer or decompile any Confidential Information."
simp = "Receiving Party can reverse engineer any objects which embody Disclosing Party's Confidential Information."

print("--- GUARDRAILS CHECK ---")
gr = LegalGuardrails.verify_all_guardrails(orig, simp, domain="auto")
pprint.pprint(gr)

print("\n--- DEONTIC POLARITY CHECK ---")
v, msg, pen = LegalGuardrails.check_deontic_polarity(orig, simp)
print("Deontic check:", v, msg, pen)

print("\n--- FULL EVALUATION ---")
res = engine.evaluate_pair(orig, simp)
print("Score:", res["score"])
print("Tier:", res["tier"])
print("Alerts:", res["guardrails"]["alerts"])
print("NLI:", res["nli"])
print("ColBERT MaxSim:", res["metrics"]["colbert_maxsim"])
print("Cosine Sim:", res["metrics"]["cosine_similarity"])
