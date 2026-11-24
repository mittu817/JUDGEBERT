import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_app.model_engine import JudgeBertEngine

engine = JudgeBertEngine()

# Test 1: MILDSum Section 37 NDPS Act Indian Court Judgment
orig1 = (
    "Section 37 states that bail should not be granted to an accused unless "
    "the accused is able to satisfy twin conditions i.e. reasonable ground for "
    "believing that the accused is not guilty of such an offence and that the accused "
    "would not commit an offence or is not likely to commit an offence, if granted bail."
)
simp1 = (
    "धारा 37 में कहा गया है कि किसी अभियुक्त को जमानत तब तक नहीं दी जानी चाहिए "
    "जब तक कि अभियुक्त दोहरी शर्तों को पूरा करने में सक्षम न हो, यानी यह विश्वास "
    "करने के लिए उचित आधार हो कि अभियुक्त इस तरह के अपराध का दोषी नहीं है और जमानत "
    "मिलने पर अभियुक्त ऐसा अपराध नहीं करेगा या ऐसा करने की संभावना नहीं है।"
)
res1 = engine.evaluate_pair(orig1, simp1)
print(f"Test 1 - MILDSum Court Judgment Equivalence:")
print(f"  Score:  {res1['score']}/10")
print(f"  Domain: {res1['domain']}")
print(f"  Mode:   {res1['mode']}")
print(f"  Passed: {res1['guardrails']['passed']}")

# Test 2: MILDSum Contradiction (Payment Ordered vs Claim Dismissed)
orig2 = (
    "The Delhi High Court on Tuesday directed the Delhi Government to pay over "
    "Rs. 16 lakhs to a senior judicial officer as reimbursement for the expenses "
    "incurred by him for his COVID-19 treatment."
)
simp2 = (
    "दिल्ली हाईकोर्ट ने दिल्ली सरकार को न्यायिक अधिकारी के COVID-19 उपचार के "
    "खर्चों की प्रतिपूर्ति के रूप में 16 लाख रुपये का भुगतान करने के दावे को खारिज कर दिया।"
)
res2 = engine.evaluate_pair(orig2, simp2)
print(f"\nTest 2 - MILDSum Directive Contradiction (Reimbursement Directive vs Claim Dismissed):")
print(f"  Score:  {res2['score']}/10")
print(f"  Domain: {res2['domain']}")
print(f"  Alerts: {[a['message'] for a in res2['guardrails']['alerts']]}")
print(f"  Passed: {res2['guardrails']['passed']}")

# Test 3: Benchmark examples count
bench = engine.get_benchmark_examples()
mildsum_items = [b for b in bench if "MILDSum" in b["title"] or "MILDSum" in b.get("category", "")]
print(f"\nTest 3 - Benchmark Examples:")
print(f"  Total examples: {len(bench)}")
print(f"  MILDSum examples: {len(mildsum_items)} (IDs: {[b['id'] for b in mildsum_items]})")
