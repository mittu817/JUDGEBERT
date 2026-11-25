import urllib.request
import json
import re
import time

BASE_URL = "http://127.0.0.1:5000"

def test_homepage():
    print("\n--- TEST 1: Homepage & UI Assets ---")
    req = urllib.request.urlopen(f"{BASE_URL}/")
    html = req.read().decode("utf-8")
    assert req.status == 200
    assert "Single Inbuilt Unified Mode" in html
    assert "nliCard" in html
    assert "Mutual Entailment" in html
    assert "documentAuditPanel" in html
    assert "origFileInput" in html
    assert "simpFileInput" in html
    assert "downloadExportedDoc" in html
    assert "Multilingual Corpus" in html
    assert "datasetLangFilter" in html
    assert "origLangSelect" not in html
    assert "simpLangSelect" not in html
    print("✅ Homepage UI markup confirmed (clean UI without manual language dropdowns).")

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

def test_auto_repair(case_name, orig, flawed, domain="auto", target_lang="auto", expected_min_score=8.0):
    time.sleep(4.5)
    print(f"\n--- REPAIR TEST: {case_name} (Target Lang: {target_lang}) ---")
    payload = json.dumps({"original": orig, "simplification": flawed, "domain": domain, "target_lang": target_lang}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/repair",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read().decode("utf-8"))

    print(f"Target Lang:         {data.get('target_lang', 'auto')}")
    print(f"Initial Score:       {data['initial_score']}/10")
    print(f"Repaired Score:      {data['repaired_score']}/10 ({data['score_delta']})")
    print(f"Repaired Text:       {data['repaired_target']}")
    print(f"Actions Taken:       {data['actions']}")
    print(f"Guardrails Cleared:  {data['evaluation']['guardrails']['passed']}")

    assert data["success"] is True, "Repair failed"
    assert data["repaired_score"] >= expected_min_score, f"Expected repaired score >= {expected_min_score}, got {data['repaired_score']}"
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

    # 1b. General Scope Divergence (User Case: Family vs Everyone)
    g1b = test_evaluation(
        "[General Context] Scope Divergence (Family vs Everyone)",
        "I love my family",
        "I love everyone"
    )
    assert g1b["score"] <= 5.5, f"Expected score <= 5.5 for unentailed scope drift, got {g1b['score']}"
    assert "EN↔EN" in g1b["mode"], f"Expected EN↔EN mode, got {g1b['mode']}"
    assert g1b["guardrails"]["passed"] is False, "Expected guardrails failure for scope divergence"
    print("✅ General scope divergence correctly penalized (EN↔EN detected, score <= 5.5).")

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

    # 5b. Cross-Lingual EN -> HI Legal Clause
    g5b = test_evaluation(
        "[Cross-Lingual EN -> HI] Legal Lease Rent Obligation",
        "The tenant must pay rent on the first day of each month.",
        "किराएदार को हर महीने की पहली तारीख को किराया देना होगा।"
    )
    assert g5b["score"] >= 7.5
    assert "HI" in g5b["mode"] or "Hindi" in g5b["mode"]

    # 5c. Hindi Legal Prohibition Reversal Contradiction
    g5c = test_evaluation(
        "[Hindi Legal Contradiction] Prohibition vs Permission Inversion",
        "कर्मचारियों को ग्राहकों के पासवर्ड साझा करने की सख्त मनाही है।",
        "कर्मचारी आवश्यकता पड़ने पर ग्राहकों के पासवर्ड साझा कर सकते हैं。"
    )
    assert g5c["score"] <= 1.0
    assert g5c["guardrails"]["passed"] is False

    # 5d. MILDSum Cross-Lingual Indian Legal Judgment Equivalence
    g5d = test_evaluation(
        "[MILDSum Indian Legal] High Court Section 37 NDPS Act Statutory Ruling",
        "Section 37 states that bail should not be granted to an accused unless the accused is able to satisfy twin conditions i.e. reasonable ground for believing that the accused is not guilty of such an offence and that the accused would not commit an offence or is not likely to commit an offence, if granted bail.",
        "धारा 37 में कहा गया है कि किसी अभियुक्त को जमानत तब तक नहीं दी जानी चाहिए जब तक कि अभियुक्त दोहरी शर्तों को पूरा करने में सक्षम न हो, यानी यह विश्वास करने के लिए उचित आधार हो कि अभियुक्त इस तरह के अपराध का दोषी नहीं है और जमानत मिलने पर अभियुक्त ऐसा अपराध नहीं करेगा या ऐसा करने की संभावना नहीं है।"
    )
    assert g5d["score"] >= 7.0
    assert g5d["guardrails"]["passed"] is True
    assert "HI" in g5d["mode"] or "Hindi" in g5d["mode"]

    # 5e. MILDSum Court Directive Reversal Contradiction
    g5e = test_evaluation(
        "[MILDSum Contradiction] High Court Directive Inversion (Payment Ordered vs Claim Dismissed)",
        "The Delhi High Court on Tuesday directed the Delhi Government to pay over Rs. 16 lakhs to a senior judicial officer as reimbursement for the expenses incurred by him for his COVID-19 treatment.",
        "दिल्ली हाईकोर्ट ने दिल्ली सरकार को न्यायिक अधिकारी के COVID-19 उपचार के खर्चों की प्रतिपूर्ति के रूप में 16 लाख रुपये का भुगतान करने के दावे को खारिज कर दिया।"
    )
    assert g5e["score"] <= 1.0
    assert g5e["guardrails"]["passed"] is False

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

    # 7b. Reverse Engineering Prohibition Reversal (Forbidden -> Permitted)
    l2b = test_evaluation(
        "[Legal Contract] Reverse Engineering Prohibition Flip (Forbidden -> Permitted)",
        "The Receiving Parties undertake and agree: f) Not to alter, modify, disassemble, reverse engineer or decompile any Confidential Information.",
        "Receiving Party can reverse engineer any objects which embody Disclosing Party's Confidential Information."
    )
    assert l2b["score"] <= 1.0
    assert l2b["guardrails"]["passed"] is False
    assert any("Prohibition Reversal" in a["message"] or "Direct Meaning Inversion" in a["message"] for a in l2b["guardrails"]["alerts"])

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
    assert "apple" in r1["repaired_target"].lower() and "next" in r1["repaired_target"].lower()
    assert not any(w in r1["repaired_target"].lower() for w in ["next acquired apple", "next purchased apple", "next bought apple"])

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
    assert "Receiving Party" in r4["repaired_target"]
    assert any(w in r4["repaired_target"].lower() for w in ["return", "send back", "hand back", "give back", "restituer"])

    # 14. Legal Deontic Polarity Reversal Repair
    r5 = test_auto_repair(
        "[Auto-Repair] Legal Insurance Deontic Flip (FR)",
        "L'assuré doit aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        "L'assuré ne doit pas aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        domain="legal"
    )
    assert "ne doit pas" not in r5["repaired_target"]
    assert any(w in r5["repaired_target"].lower() for w in ["doit", "devez", "doivent", "obligation", "tenu", "informer", "aviser", "déclarer", "notifier", "assureur"])

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
    assert any(w in r7["repaired_target"].lower() for w in ["deduction", "subtraction", "withholding", "reducing", "full rent", "reminder", "demand", "bill", "request", "notice", "in full"])
    assert r7["repaired_target"].strip().lower() != "The tenant shall pay rent on the first of each month without demand or deduction.".lower()

    # 17. Prohibition vs Permission Flip (Password)
    r8 = test_auto_repair(
        "[Auto-Repair] Prohibition Polarity Restored in Simplification",
        "Employees are strictly prohibited from sharing customer passwords under any circumstances.",
        "Employees can share customer passwords if needed.",
        domain="auto"
    )
    assert any(p in r8["repaired_target"].lower() for p in ["must not share", "cannot share", "not allowed to share", "prohibited", "forbidden", "barred", "never share"])
    assert r8["repaired_score"] >= 8.5
    assert r8["repaired_target"].strip().lower() != "Employees are strictly prohibited from sharing customer passwords under any circumstances.".lower()

    # ==========================================
    # PART D: TARGET LANGUAGE SIMPLIFICATION TESTS (Hindi, French, English)
    # ==========================================
    print("\n" + "="*80)
    print("PART D: TARGET LANGUAGE SELECTION TESTS (Hindi, French, English)")
    print("="*80)

    # 18. Target Language: Hindi (Devanagari)
    r_hi = test_auto_repair(
        "[Multilingual Target] English Clause -> Target Simplification in Hindi",
        "The tenant must pay rent on the first day of each month.",
        "The tenant is not required to pay rent.",
        domain="legal",
        target_lang="hi",
        expected_min_score=7.8
    )
    assert re.search(r"[\u0900-\u097F]", r_hi["repaired_target"]), "Output must be in Devanagari Hindi"
    assert r_hi["target_lang"] == "hi"
    print("✅ Target simplification in certified Hindi verified!")

    # 19. Target Language: French
    r_fr = test_auto_repair(
        "[Multilingual Target] English Clause -> Target Simplification in French",
        "Employees are strictly prohibited from sharing customer passwords under any circumstances.",
        "Employees can share passwords.",
        domain="legal",
        target_lang="fr",
        expected_min_score=8.0
    )
    assert any(w in r_fr["repaired_target"].lower() for w in ["interdit", "ne doivent pas", "ne doit pas", "aucun cas", "partager"]), "Output must be in French"
    assert r_fr["target_lang"] == "fr"
    print("✅ Target simplification in certified French verified!")

    # 20. Target Language: English
    r_en = test_auto_repair(
        "[Multilingual Target] French Clause -> Target Simplification in English",
        "L'assuré doit aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        "L'assuré ne doit pas aviser l'assureur.",
        domain="legal",
        target_lang="en",
        expected_min_score=8.0
    )
    assert any(w in r_en["repaired_target"].lower() for w in ["must", "inform", "notify", "insurer", "days", "claim"]), "Output must be in Plain English"
    assert r_en["target_lang"] == "en"
    print("✅ Target simplification in Plain English verified!")

    # ============================================================================
    # PART E: SMART AI LANGUAGE DETECTION & LEGAL LATIN TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("PART E: SMART AI LANGUAGE DETECTION & LEGAL LATIN TESTS")
    print("="*80)

    # 21. Automatic Backend Language Detection: Legal Latin as English (force majeure)
    latin_payload = json.dumps({
        "original": "The supplier shall be excused from liability in the event of force majeure or acts of God.",
        "simplification": "In cases of force majeure, the supplier is not liable."
    }).encode("utf-8")
    req_latin = urllib.request.Request(f"{BASE_URL}/api/evaluate", data=latin_payload, headers={"Content-Type": "application/json"})
    d_latin = json.loads(urllib.request.urlopen(req_latin).read().decode("utf-8"))
    assert d_latin["lang_orig"] == "en", f"Expected 'en', got {d_latin.get('lang_orig')}"
    assert d_latin["lang_simp"] == "en", f"Expected 'en', got {d_latin.get('lang_simp')}"
    assert d_latin["domain"] == "legal", f"Expected 'legal' domain for force majeure, got {d_latin.get('domain')}"
    assert "EN↔EN" in d_latin["mode"]
    print("✅ Backend automatically detected Legal Latin (force majeure) as English in Legal Context (EN↔EN)!")

    # 22. Automatic Backend Detection of Hindi (HI↔HI)
    hi_payload = json.dumps({
        "original": "किराएदार को हर महीने की पहली तारीख को किराया देना अनिवार्य है।",
        "simplification": "किराएदार को हर महीने समय पर किराया देना होगा।"
    }).encode("utf-8")
    req_hi = urllib.request.Request(f"{BASE_URL}/api/evaluate", data=hi_payload, headers={"Content-Type": "application/json"})
    d_hi = json.loads(urllib.request.urlopen(req_hi).read().decode("utf-8"))
    assert d_hi["lang_orig"] == "hi", f"Expected 'hi', got {d_hi.get('lang_orig')}"
    assert d_hi["lang_simp"] == "hi", f"Expected 'hi', got {d_hi.get('lang_simp')}"
    assert "HI↔HI" in d_hi["mode"]
    print("✅ Backend automatically detected Hindi clauses as HI↔HI!")

    # 23. Automatic Backend Detection of French (FR↔FR)
    fr_payload = json.dumps({
        "original": "L'assuré doit aviser l'assureur de tout sinistre dans un délai de cinq jours.",
        "simplification": "L'assuré informe l'assureur dans un délai de 5 jours."
    }).encode("utf-8")
    req_fr = urllib.request.Request(f"{BASE_URL}/api/evaluate", data=fr_payload, headers={"Content-Type": "application/json"})
    d_fr = json.loads(urllib.request.urlopen(req_fr).read().decode("utf-8"))
    assert d_fr["lang_orig"] == "fr", f"Expected 'fr', got {d_fr.get('lang_orig')}"
    assert d_fr["lang_simp"] == "fr", f"Expected 'fr', got {d_fr.get('lang_simp')}"
    assert "FR↔FR" in d_fr["mode"]
    print("✅ Backend automatically detected French clauses as FR↔FR!")

    # ============================================================================
    # PART F: PRIVACY SHIELD (LOCAL PII MASKING BEFORE GEMINI CLOUD CALLS)
    # ============================================================================
    print("\n" + "="*80)
    print("PART F: PRIVACY SHIELD (LOCAL PII MASKING & RESTORATION TESTS)")
    print("="*80)

    # 24. Check /api/gemini-status exposes active privacy masking
    g_status = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/gemini-status").read().decode("utf-8"))
    assert g_status.get("privacy_masking") is True, f"Expected privacy_masking True, got {g_status}"
    print("✅ Backend status confirms Privacy Shield is ACTIVE!")

    # 25. Verify secrets are redacted from Gemini prompt and faithfully restored in EN -> EN
    from demo_app.privacy_masker import PrivacyMasker
    conf_orig = "Acme Holdings LLC shall pay Mr. Rajesh Sharma $2,500,000 (15% bonus) via billing@acmecorp.com by December 15, 2026."
    conf_flawed = "Acme Holdings LLC is not required to pay $100,000 to Mr. Rajesh Sharma by December 15, 2026."

    m_orig, m_simp, _ma, sess = PrivacyMasker.mask_inputs(conf_orig, conf_flawed, [])
    for secret in ["Acme Holdings LLC", "Rajesh Sharma", "$2,500,000", "15%", "billing@acmecorp.com", "December 15, 2026", "$100,000"]:
        assert secret not in m_orig and secret not in m_simp, f"Leak detected! {secret} found in masked payload!"
    print(f"✅ Pre-flight scrub verified ({sess.masked_count} entities redacted: {sess.summary_description()})")

    r_priv = test_auto_repair(
        "[Privacy Shield] Confidential NDA/Payment Auto-Repair (EN -> EN)",
        conf_orig,
        conf_flawed,
        domain="legal",
        target_lang="en",
        expected_min_score=8.0
    )
    repaired_priv = r_priv["repaired_target"]
    for must_keep in ["Acme Holdings LLC", "Rajesh Sharma", "$2,500,000", "15%", "billing@acmecorp.com", "December 15, 2026"]:
        assert must_keep in repaired_priv, f"Expected unmasked secret '{must_keep}' in repaired output: {repaired_priv}"
    assert not re.search(r"\[(?:ORG|PERSON|AMOUNT|PERCENT|EMAIL|DATE|NUM)_\d+\]", repaired_priv), f"Leaked placeholder tag in: {repaired_priv}"
    if r_priv.get("generator") == "gemini":
        assert r_priv.get("privacy_shield", {}).get("masked_count", 0) >= 6
        assert any("Privacy Shield" in act for act in r_priv.get("actions", []))
    print("✅ Confidential contract restored 100% locally with zero placeholder leakage!")

    # 26. Verify cross-lingual Privacy Shield (EN -> HI)
    r_priv_hi = test_auto_repair(
        "[Privacy Shield] Confidential Contract Cross-Lingual (EN -> HI)",
        "Acme Holdings LLC must pay $2,500,000 by 15/12/2026.",
        "Acme Holdings LLC does not have to pay $100,000.",
        domain="legal",
        target_lang="hi",
        expected_min_score=7.5
    )
    rep_hi = r_priv_hi["repaired_target"]
    assert re.search(r"[\u0900-\u097F]", rep_hi), f"Expected Devanagari Hindi in: {rep_hi}"
    assert "Acme Holdings LLC" in rep_hi and "$2,500,000" in rep_hi and "15/12/2026" in rep_hi, f"Missing restored entity in Hindi output: {rep_hi}"
    assert not re.search(r"\[(?:ORG|PERSON|AMOUNT|DATE|NUM)_\d+\]", rep_hi), f"Leaked placeholder in Hindi output: {rep_hi}"
    print("✅ Cross-lingual Privacy Shield (EN -> HI) verified!")

    # ============================================================================
    # PART G: WHOLE-DOCUMENT MULTI-CLAUSE & COREFERENCE RESOLUTION TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("PART G: WHOLE-DOCUMENT MULTI-CLAUSE AUDIT & COREFERENCE TESTS")
    print("="*80)

    # 27. Multi-clause document with alias ("Vendor"), pronoun ("They"), contradiction, and omitted clause
    doc_orig = (
        'Acme Holdings LLC ("Vendor") agrees to provide cloud infrastructure services starting January 1, 2026.\n'
        'Vendor shall pay a 15% penalty if downtime exceeds 4 hours in a billing month.\n'
        'They must deliver all monthly security audits within 30 days after month end.\n'
        'All intellectual property created during the term remains exclusively owned by the Client.'
    )
    doc_simp = (
        'Acme Holdings LLC will provide cloud hosting services beginning January 1, 2026.\n'
        'Acme Holdings LLC has to pay a 15% penalty when downtime is more than 4 hours per month.\n'
        'Acme Holdings LLC is not required to deliver monthly security audits.'
    )

    doc_payload = json.dumps({"original": doc_orig, "simplification": doc_simp, "domain": "legal"}).encode("utf-8")
    req_doc = urllib.request.Request(f"{BASE_URL}/api/evaluate", data=doc_payload, headers={"Content-Type": "application/json"})
    d_doc = json.loads(urllib.request.urlopen(req_doc).read().decode("utf-8"))

    assert d_doc.get("is_document") is True, "Expected Whole-Document mode to engage automatically"
    assert d_doc["summary"]["total_orig_clauses"] == 4
    assert d_doc["summary"]["preserved"] >= 2, f"Expected first 2 clauses preserved after Vendor resolution, got {d_doc['summary']}"
    assert d_doc["summary"]["contradictions"] >= 1, "Expected Clause 3 obligation flip contradiction"
    assert d_doc["summary"]["omitted"] >= 1, "Expected Clause 4 IP omission detected"
    assert d_doc["clauses"][2]["status"] == "contradiction", f"Expected Clause 3 contradiction, got {d_doc['clauses'][2]}"
    assert d_doc["clauses"][3]["status"] == "omitted", f"Expected Clause 4 omitted, got {d_doc['clauses'][3]}"
    assert any("Vendor" in r or "They" in r for r in d_doc.get("resolved_references", [])), f"Expected alias/pronoun resolution, got {d_doc.get('resolved_references')}"
    assert d_doc["score"] <= 4.5, f"Expected document risk score <= 4.5, got {d_doc['score']}"
    print(f"✅ Whole-Document Risk Audit verified (Score: {d_doc['score']}/10, Preserved: {d_doc['summary']['preserved']}, Contradictions: {d_doc['summary']['contradictions']}, Omitted: {d_doc['summary']['omitted']})")

    # 28. Faithful multi-clause document with pronoun & alias resolution via /api/evaluate-document
    doc_clean_simp = (
        'Acme Holdings LLC will provide cloud hosting services beginning January 1, 2026.\n'
        'Acme Holdings LLC has to pay a 15% penalty when downtime is more than 4 hours per month.\n'
        'Acme Holdings LLC must submit all monthly security audits within 30 days after the end of the month.\n'
        'All intellectual property created during the contract term stays solely owned by the Client.'
    )
    req_clean = urllib.request.Request(
        f"{BASE_URL}/api/evaluate-document",
        data=json.dumps({"original": doc_orig, "simplification": doc_clean_simp, "domain": "legal"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    d_clean = json.loads(urllib.request.urlopen(req_clean).read().decode("utf-8"))
    assert d_clean.get("is_document") is True
    assert d_clean["summary"]["omitted"] == 0
    assert d_clean["summary"]["contradictions"] == 0
    assert d_clean["summary"]["preserved"] == 4, f"Expected all 4 clauses preserved, got {d_clean['clauses']}"
    assert d_clean["score"] >= 8.5, f"Expected compliant document score >= 8.5, got {d_clean['score']}"
    print(f"✅ Conforming Multi-Clause Contract verified via /api/evaluate-document (Score: {d_clean['score']}/10, 4/4 Clauses Preserved)!")

    # 29. Per-clause Auto-Repair using Document Memory's resolved antecedent ("They" -> "Acme Holdings LLC")
    c3 = d_doc["clauses"][2]
    r_doc_c3 = test_auto_repair(
        "[Document Clause Repair] Resolved Pronoun Clause #3 -> Auto-Repair",
        c3["resolved_orig"],
        c3["resolved_simp"],
        domain="legal",
        target_lang="en",
        expected_min_score=8.5
    )
    assert "Acme Holdings LLC" in r_doc_c3["repaired_target"]
    assert "30" in r_doc_c3["repaired_target"]
    print("✅ Per-clause inline repair preserved resolved antecedent ('Acme Holdings LLC') and 30-day deadline!")

    # ============================================================================
    # PART H: FILE UPLOAD EXTRACTION (.TXT, .DOCX, .PDF) & DOCUMENT AUDIT TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("PART H: DOCUMENT FILE UPLOAD (.TXT, .DOCX, .PDF) EXTRACTION TESTS")
    print("="*80)

    import io
    import uuid
    import docx as py_docx

    def upload_multipart(filename, file_bytes, content_type):
        boundary = f"----JudgeBertTest{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/api/extract-document",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

    # 30. .TXT Document Upload
    txt_bytes = doc_orig.encode("utf-8")
    res_txt = upload_multipart("msa_original.txt", txt_bytes, "text/plain")
    assert res_txt["success"] is True and res_txt["clause_count"] == 4
    print(f"✅ .TXT upload extracted {res_txt['clause_count']} clauses ({res_txt['filename']}).")

    # 31. .DOCX Document Upload
    docx_buf = io.BytesIO()
    d_obj = py_docx.Document()
    for line in doc_clean_simp.splitlines():
        d_obj.add_paragraph(line)
    d_obj.save(docx_buf)
    res_docx = upload_multipart(
        "msa_simplified.docx",
        docx_buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert res_docx["success"] is True and res_docx["clause_count"] == 4
    print(f"✅ .DOCX upload extracted {res_docx['clause_count']} clauses ({res_docx['filename']}).")

    # 32. End-to-End Whole-Document Evaluation using extracted .TXT + .DOCX texts
    res_eval_upload = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{BASE_URL}/api/evaluate-document",
        data=json.dumps({"original": res_txt["text"], "simplification": res_docx["text"], "domain": "legal"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )).read().decode("utf-8"))
    assert res_eval_upload["score"] >= 8.5 and res_eval_upload["summary"]["preserved"] == 4
    print(f"✅ Uploaded .TXT vs .DOCX Full Contract Audit passed (Score: {res_eval_upload['score']}/10, 4/4 Clauses Preserved)!")

    # 33. .PDF Upload Parser Check (Valid minimal PDF binary stream with literal text)
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>endobj\n"
        b"4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
        b"5 0 obj<< /Length 79 >>stream\n"
        b"BT /F1 12 Tf 72 720 Td (The tenant must pay rent every month.) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n0000000309 00000 n \n"
        b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n438\n%%EOF\n"
    )
    res_pdf = upload_multipart("lease_clause.pdf", minimal_pdf, "application/pdf")
    assert res_pdf["success"] is True and "tenant must pay rent" in res_pdf["text"].lower()
    print(f"✅ .PDF upload extracted text cleanly: '{res_pdf['text']}'")

    # 34. Multi-line noisy PDF w/ watermark, hyphenation break, and Page footer
    noisy_stream = (
        b"BT /F1 12 Tf 72 750 Td (CONFIDENTIAL) Tj "
        b"0 -30 Td (The contractor must sign the agree-) Tj "
        b"0 -18 Td (ment within 30 days.) Tj "
        b"0 -650 Td (Page 1 of 2) Tj ET"
    )
    noisy_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>endobj\n"
        b"4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
        b"5 0 obj<< /Length " + str(len(noisy_stream)).encode() + b" >>stream\n" + noisy_stream + b"\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    res_noisy_pdf = upload_multipart("noisy_contract.pdf", noisy_pdf, "application/pdf")
    assert res_noisy_pdf["success"] is True
    assert "CONFIDENTIAL" not in res_noisy_pdf["text"]
    assert "Page 1 of 2" not in res_noisy_pdf["text"]
    assert "agreement within 30 days." in res_noisy_pdf["text"], f"Expected de-hyphenated/unwrapped text, got: {res_noisy_pdf['text']}"
    print(f"✅ Noisy PDF header/footer/hyphenation cleaned cleanly: '{res_noisy_pdf['text']}'")

    # 35. Noisy DOCX w/ merged cells & footer rows stripped
    noisy_docx_buf = io.BytesIO()
    nd = py_docx.Document()
    nd.add_paragraph("CONFIDENTIAL")
    nd.add_paragraph("Vendor shall pay all fees monthly.")
    nt = nd.add_table(rows=2, cols=3)
    nt.cell(0, 0).merge(nt.cell(0, 1))
    nt.cell(0, 0).text = "Notice Period"
    nt.cell(0, 2).text = "30 days written notice"
    nt.cell(1, 0).text = "Page 1 of 3"
    nt.cell(1, 1).text = "DRAFT"
    nt.cell(1, 2).text = ""
    nd.save(noisy_docx_buf)
    res_noisy_docx = upload_multipart(
        "noisy_table.docx",
        noisy_docx_buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert res_noisy_docx["success"] is True
    assert "CONFIDENTIAL" not in res_noisy_docx["text"]
    assert "Page 1 of 3" not in res_noisy_docx["text"]
    assert "Notice Period: 30 days written notice." in res_noisy_docx["text"]
    assert res_noisy_docx["text"].count("Notice Period") == 1
    print(f"✅ Merged-cell DOCX & header/footer boilerplate cleaned: {res_noisy_docx['text']!r}")

    # ============================================================================
    # PART I: REPAIRED DOCUMENT EXPORT (.DOCX & .PDF) & ROUNDTRIP TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("PART I: REPAIRED DOCUMENT DOWNLOAD (.DOCX & .PDF) EXPORT TESTS")
    print("="*80)

    export_sample = (
        "Acme Holdings LLC must pay $2,500,000 by December 15, 2026.\n"
        "L'assuré doit aviser l'assureur dans un délai de 5 jours.\n"
        "किराएदार को हर महीने की पहली तारीख को किराया देना होगा।"
    )

    # 36. Export to .DOCX and roundtrip verify
    req_exp_docx = urllib.request.Request(
        f"{BASE_URL}/api/export-document",
        data=json.dumps({
            "text": export_sample,
            "format": "docx",
            "filename": "test_repaired",
            "score": 9.8,
            "tier": "Excellent Document Fidelity"
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    resp_docx_dl = urllib.request.urlopen(req_exp_docx)
    dl_docx_bytes = resp_docx_dl.read()
    assert resp_docx_dl.status == 200
    assert dl_docx_bytes[:2] == b"PK", "Exported DOCX must be a valid Office Open XML ZIP archive"
    assert "test_repaired.docx" in resp_docx_dl.headers.get("Content-Disposition", "")
    rt_docx = upload_multipart("roundtrip_repaired.docx", dl_docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert "$2,500,000" in rt_docx["text"] and "किराएदार" in rt_docx["text"]
    print(f"✅ Repaired .DOCX download verified ({len(dl_docx_bytes)} bytes, multilingual roundtrip OK).")

    # 37. Export to .PDF and roundtrip verify
    req_exp_pdf = urllib.request.Request(
        f"{BASE_URL}/api/export-document",
        data=json.dumps({
            "text": export_sample,
            "format": "pdf",
            "filename": "test_repaired",
            "score": 9.8,
            "tier": "Excellent Document Fidelity"
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    resp_pdf_dl = urllib.request.urlopen(req_exp_pdf)
    dl_pdf_bytes = resp_pdf_dl.read()
    assert resp_pdf_dl.status == 200
    assert dl_pdf_bytes.startswith(b"%PDF-"), "Exported PDF must start with %PDF-"
    assert "test_repaired.pdf" in resp_pdf_dl.headers.get("Content-Disposition", "")
    rt_pdf = upload_multipart("roundtrip_repaired.pdf", dl_pdf_bytes, "application/pdf")
    assert "$2,500,000" in rt_pdf["text"] and "assuré" in rt_pdf["text"] and "किराएदार" in rt_pdf["text"]
    print(f"✅ Repaired .PDF download verified ({len(dl_pdf_bytes)} bytes, EN/FR/HI Unicode roundtrip OK).")

    # ============================================================================
    # PART J: MULTILINGUAL CORPUS EXPLORER TESTS (ENGLISH, HINDI & FRENCH)
    # ============================================================================
    print("\n" + "="*80)
    print("PART J: MULTILINGUAL CORPUS EXPLORER API TESTS (EN + HI + FR)")
    print("="*80)

    # 38. Stats endpoint includes FR, EN, HI counts
    stats = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/stats").read().decode("utf-8"))
    assert stats.get("fr_count", 0) > 1000, f"Expected FrJUDGE French count > 1000, got {stats}"
    assert stats.get("en_count", 0) > 100, f"Expected ContractNLI English count > 100, got {stats}"
    assert stats.get("hi_count", 0) > 100, f"Expected MILDSum/Hindi count > 100, got {stats}"
    print(f"✅ Corpus Stats verified: Total={stats['total_annotations']} (EN={stats['en_count']}, HI={stats['hi_count']}, FR={stats['fr_count']})")

    # 39. Default mixed browse includes all 3 languages (EN, HI, FR)
    ds_all = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/dataset?lang=all&limit=30").read().decode("utf-8"))
    langs_seen = {it.get("lang") for it in ds_all}
    assert {"en", "hi", "fr"}.issubset(langs_seen), f"Expected all 3 languages in default browse, got {langs_seen}"
    print(f"✅ Default corpus browse showcases all 3 languages evenly ({langs_seen}).")

    # 40. Filter by English, Hindi, and French individually
    ds_en = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/dataset?lang=en&limit=10").read().decode("utf-8"))
    assert len(ds_en) == 10 and all("en" in (it.get("langs") or [it.get("lang")]) for it in ds_en)
    ds_hi = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/dataset?lang=hi&limit=10").read().decode("utf-8"))
    assert len(ds_hi) == 10 and any(re.search(r"[\u0900-\u097F]", it["simplification"]) for it in ds_hi)
    ds_fr = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/dataset?lang=fr&limit=10").read().decode("utf-8"))
    assert len(ds_fr) == 10 and all(it.get("lang") == "fr" for it in ds_fr)
    print("✅ Per-language corpus filtering (EN=ContractNLI, HI=MILDSum/Hindi, FR=FrJUDGE) verified!")

    print("\n" + "="*80)
    print(" ALL EVALUATION & CLOSED-LOOP MULTILINGUAL AUTO-REPAIR TESTS PASSED WITH 100% ACCURACY! ")
    print("="*80)

