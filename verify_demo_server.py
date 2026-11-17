import urllib.request
import json

BASE_URL = "http://127.0.0.1:5000"

def test_homepage():
    print("\n--- TEST 1: Homepage & UI Assets ---")
    req = urllib.request.urlopen(f"{BASE_URL}/")
    html = req.read().decode("utf-8")
    assert req.status == 200
    assert "domainSelector" in html
    assert "nliCard" in html
    assert "Mutual Entailment" in html
    print("✅ Homepage UI markup confirmed with Domain Selector and Bidirectional NLI card.")

def test_examples_api():
    print("\n--- TEST 2: Benchmark Examples API ---")
    req = urllib.request.urlopen(f"{BASE_URL}/api/examples")
    items = json.loads(req.read().decode("utf-8"))
    print(f"Total benchmark examples loaded: {len(items)}")
    for ex in items[:8]:
        print(f" - [{ex['id']}] {ex['title']}")
    assert len(items) >= 10
    print("✅ Benchmark examples API verified (includes both General Context & Legal).")

def test_evaluation(case_name, orig, simp, domain="auto"):
    print(f"\n--- TEST: {case_name} ---")
    payload = json.dumps({"original": orig, "simplification": simp, "domain": domain}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/evaluate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode("utf-8"))
    
    print(f"Domain:              {data.get('domain', 'N/A')}")
    print(f"Score:               {data['score']}/10 ({data['tier']})")
    print(f"NLI Verdict:         {data['nli']['verdict']}")
    print(f"ColBERT MaxSim F1:   {data['metrics'].get('colbert_maxsim', data['metrics']['cosine_similarity'])}% (Fwd: {data['metrics'].get('colbert_forward', '--')}%, Rev: {data['metrics'].get('colbert_reverse', '--')}%)")
    print(f"Guardrails Passed:   {data['guardrails']['passed']}")
    if data['guardrails']['alerts']:
        for a in data['guardrails']['alerts']:
            print(f"  -> [{a['level'].upper()}] {a['message']}")
    return data

def test_auto_repair(case_name, orig, flawed, domain="auto", expected_min_score=9.0):
    print(f"\n--- REPAIR TEST: {case_name} ---")
    payload = json.dumps({"original": orig, "simplification": flawed, "domain": domain}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/repair",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode("utf-8"))

    print(f"Initial Score:       {data['initial_score']}/10")
    print(f"Repaired Score:      {data['repaired_score']}/10 ({data['score_delta']})")
    print(f"Repaired Text:       {data['repaired_target']}")
    print(f"Actions Taken:       {data['actions']}")
    print(f"Guardrails Cleared:  {data['evaluation']['guardrails']['passed']}")

    assert data["success"] is True, "Repair failed"
    assert data["repaired_score"] >= expected_min_score, f"Expected repaired score >= {expected_min_score}, got {data['repaired_score']}"
    assert data["repaired_score"] > data["initial_score"], "Repaired score did not improve"
    assert data["evaluation"]["guardrails"]["passed"] is True, "Guardrails did not pass on repaired output"
    return data

if __name__ == "__main__":
    test_homepage()
    test_examples_api()
    
    # ==========================================
    # PART A: GENERAL CONTEXT MEANING PRESERVATION
    # ==========================================
    print("\n" + "="*80)
    print("PART A: GENERAL CONTEXT MEANING PRESERVATION TESTS")
    print("="*80)

    # 1. General Science Factually Equivalent Paraphrase
    g1 = test_evaluation(
        "[General Context] Science Paraphrase (NASA James Webb)",
        "NASA launched the James Webb Space Telescope in December 2021 to observe the earliest galaxies formed after the Big Bang.",
        "In December 2021, NASA sent the James Webb Space Telescope into space to study the universe's first galaxies."
    )
    assert g1["score"] >= 8.5
    assert g1["domain"] == "general"

    # 2. General Entity / Actor Inversion
    g2 = test_evaluation(
        "[General Context] Entity Inversion (Apple vs NeXT)",
        "In 1996, Apple acquired NeXT for 429 million dollars to replace its aging operating system.",
        "In 1996, NeXT acquired Apple for 429 million dollars to replace its aging operating system."
    )
    assert g2["score"] <= 1.0
    assert any("Inversion d'acteurs" in a["message"] for a in g2["guardrails"]["alerts"])

    # 3. General Causal Direction Reversal
    g3 = test_evaluation(
        "[General Context] Causal Reversal (Match vs Thunderstorm)",
        "The outdoor soccer match was cancelled because a severe thunderstorm hit the city.",
        "A severe thunderstorm hit the city because the outdoor soccer match was cancelled."
    )
    assert g3["score"] <= 1.0
    assert any("Inversion de causalité" in a["message"] for a in g3["guardrails"]["alerts"])

    # 4. General Quantitative / Date Alteration
    g4 = test_evaluation(
        "[General Context] Quantitative Alteration (Eiffel Tower)",
        "The Eiffel Tower was completed in 1889 and stands approximately 330 meters tall.",
        "The Eiffel Tower was built in 1989 and is around 130 meters tall."
    )
    assert g4["score"] <= 2.0
    assert any("Modification de délai" in a["message"] or "altéré" in a["message"] or not g4["guardrails"]["passed"] for a in g4["guardrails"]["alerts"])

    # 5. General Cross-Lingual FR -> EN
    g5 = test_evaluation(
        "[General Cross-Lingual] Louvre Museum Visitors (FR -> EN)",
        "Le musée du Louvre a accueilli près de 8,9 millions de visiteurs l'année dernière à Paris.",
        "The Louvre museum in Paris welcomed nearly 8.9 million visitors last year."
    )
    assert g5["score"] >= 8.5
    assert g5["guardrails"]["passed"] is True

    # ==========================================
    # PART B: LEGAL & CONTRACTUAL MEANING PRESERVATION
    # ==========================================
    print("\n" + "="*80)
    print("PART B: LEGAL & CONTRACTUAL MEANING PRESERVATION TESTS")
    print("="*80)

    # 6. English Legal Party Inversion
    l1 = test_evaluation(
        "[Legal Contract] Party Role Swap (Receiving vs Disclosing)",
        "The Receiving Party shall return all Confidential Information to the Disclosing Party upon written request.",
        "The Disclosing Party will return all Confidential Information to the Receiving Party upon written request."
    )
    assert l1["score"] <= 1.0
    assert any("Inversion des parties" in a["message"] for a in l1["guardrails"]["alerts"])

    # 7. French Legal Party Inversion
    l2 = test_evaluation(
        "[Legal Contract] French Party Swap (Assureur vs Assuré)",
        "L'assureur doit indemniser l'assuré pour les dommages matériels directs.",
        "L'assuré doit indemniser l'assureur pour les dommages matériels directs."
    )
    assert l2["score"] <= 1.0
    assert any("Inversion des parties" in a["message"] for a in l2["guardrails"]["alerts"])

    # 8. English Legal Flipped Obligation
    l3 = test_evaluation(
        "[Legal Contract] Flipped Obligation (Shall Pay vs Not Required)",
        "The tenant shall pay rent on the first of each month without demand or deduction.",
        "The tenant is not required to pay rent on the first of the month."
    )
    assert l3["score"] <= 1.0
    assert any("Annulation d'obligation" in a["message"] for a in l3["guardrails"]["alerts"])

    # 9. Legal Cross-Lingual FR -> EN Equivalent
    l4 = test_evaluation(
        "[Legal Cross-Lingual] Subdivision C Exclusion (FR -> EN)",
        "Cette Subdivision C exclut la partie des frais médicaux visée par une autre assurance, gouvernementale ou privée, sauf si cette autre assurance est semblable à la présente.",
        "Subdivision C excludes medical expenses covered by other government or private insurance, unless that other insurance is similar to this policy."
    )
    assert l4["score"] >= 8.0
    assert l4["guardrails"]["passed"] is True

    # ==========================================
    # PART C: CLOSED-LOOP AUTO-REPAIR TESTS
    # ==========================================
    print("\n" + "="*80)
    print("PART C: CLOSED-LOOP AUTO-REPAIR (SELF-CORRECTION) ARBITER TESTS")
    print("="*80)

    # 10. General Entity Inversion Repair
    r1 = test_auto_repair(
        "[Auto-Repair] General Entity Inversion (Apple vs NeXT)",
        "In 1996, Apple acquired NeXT for 429 million dollars to replace its aging operating system.",
        "In 1996, NeXT acquired Apple for 429 million dollars to replace its aging operating system.",
        domain="general"
    )
    assert any(w in r1["repaired_target"] for w in ["Apple purchased NeXT", "Apple acquired NeXT", "Apple bought NeXT"])

    # 11. General Causal Direction Reversal Repair
    r2 = test_auto_repair(
        "[Auto-Repair] General Causal Reversal (Match vs Thunderstorm)",
        "The outdoor soccer match was cancelled because a severe thunderstorm hit the city.",
        "A severe thunderstorm hit the city because the outdoor soccer match was cancelled.",
        domain="general"
    )
    assert "outdoor soccer match" in r2["repaired_target"].lower()
    assert any(w in r2["repaired_target"].lower() for w in ["cancelled", "called off"])
    assert "thunderstorm" in r2["repaired_target"].lower()

    # 12. General Quantitative & Date Alteration Repair
    r3 = test_auto_repair(
        "[Auto-Repair] Quantitative Alteration (Eiffel Tower)",
        "The Eiffel Tower was completed in 1889 and stands approximately 330 meters tall.",
        "The Eiffel Tower was built in 1989 and is around 130 meters tall.",
        domain="general"
    )
    assert "1889" in r3["repaired_target"]
    assert "330" in r3["repaired_target"]

    # 13. Legal Contract Party Inversion Repair
    r4 = test_auto_repair(
        "[Auto-Repair] Legal Contract Party Inversion",
        "The Receiving Party shall return all Confidential Information to the Disclosing Party upon written request.",
        "The Disclosing Party will return all Confidential Information to the Receiving Party upon written request.",
        domain="legal"
    )
    assert "Receiving Party" in r4["repaired_target"][:25]

    # 14. Legal Deontic Polarity Reversal Repair
    r5 = test_auto_repair(
        "[Auto-Repair] Legal Insurance Deontic Flip (FR)",
        "L'assuré doit aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        "L'assuré ne doit pas aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        domain="legal"
    )
    assert "ne doit pas" not in r5["repaired_target"]
    assert "doit" in r5["repaired_target"]

    # 15. Legal Contract Duration Mismatch Repair
    r6 = test_auto_repair(
        "[Auto-Repair] Legal Contract Duration Mismatch",
        "The contractor must deliver the final report within 30 days following approval.",
        "The contractor must deliver the final report within 15 days following approval.",
        domain="legal"
    )
    assert "30 days" in r6["repaired_target"]

    # 16. Target Phrase Preservation with Missing Qualifier Restored
    r7 = test_auto_repair(
        "[Auto-Repair] Target Preserved & Missing Condition Injected (Tenant)",
        "The tenant shall pay rent on the first of each month without demand or deduction.",
        "The tenant must pay rent by the first of each month",
        domain="auto"
    )
    assert "without demand or deduction" in r7["repaired_target"].lower()
    assert "by the first of each month" in r7["repaired_target"].lower()
    assert r7["repaired_target"].strip().lower() != "The tenant shall pay rent on the first of each month without demand or deduction.".lower()

    # 17. Prohibition vs Permission Flip (Password)
    r8 = test_auto_repair(
        "[Auto-Repair] Prohibition Polarity Restored in Simplification",
        "Employees are strictly prohibited from sharing customer passwords under any circumstances.",
        "Employees can share customer passwords if needed.",
        domain="auto"
    )
    assert any(p in r8["repaired_target"].lower() for p in ["must not share", "cannot share", "not allowed to share", "prohibited"])
    assert r8["repaired_score"] >= 9.0
    assert r8["repaired_target"].strip().lower() != "Employees are strictly prohibited from sharing customer passwords under any circumstances.".lower()

    print("\n" + "="*80)
    print(" ALL 17 EVALUATION & CLOSED-LOOP AUTO-REPAIR TESTS PASSED WITH 100% ACCURACY! ")
    print("="*80)

