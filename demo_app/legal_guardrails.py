import re
from typing import Dict, List, Tuple, Any

class SyntacticRole:
    AGENT = "AGENT"          # Subject/performer of action or passive 'by' agent
    PATIENT = "PATIENT"      # Direct object/theme acted upon
    RECIPIENT = "RECIPIENT"  # Beneficiary / directional recipient (to, envers, à, pour)
    UNKNOWN = "UNKNOWN"

class LegalGuardrails:
    """
    Comprehensive Neuro-Symbolic Legal Guardrails for Contract & Insurance Simplification:
    1. Deontic Modals & Polarity Inversion (shall vs shall not, doit vs ne doit pas)
    2. Critical Exception & Condition Loss Detector (unless, except, sauf si)
    3. Numerical & Temporal Constraint Matching (amounts, %, durations)
    """

    # --- Deontic Patterns ---
    OBLIGATION_EN = [
        r"\bshall\b", r"\bmust\b", r"\bis required to\b", r"\bare required to\b",
        r"\bundertakes? to\b", r"\bagrees? to\b", r"\bhas an obligation to\b",
        r"\bis obligated to\b", r"\bcovenants to\b", r"\bdirected\b", r"\bordered\b",
        r"\bmandated\b", r"\bunder an obligation\b"
    ]
    PROHIBITION_EN = [
        r"\bshall not\b", r"\bmust not\b", r"\bmay not\b", r"\bcannot\b",
        r"\bprohibited from\b", r"\bforbidden to\b", r"\bnot permitted to\b",
        r"\bagrees? not to\b", r"\bundertakes? not to\b", r"\bshould not be granted\b",
        r"\bshall not be granted\b", r"\bdenied\b", r"\bdismissed\b", r"\bquashed\b",
        r"\bnot to\b", r"\brefrain from\b", r"\bno right to\b", r"\bstrictly prohibited\b",
        r"\bwill not\b", r"\bshall in no event\b", r"\bnot allowed to\b"
    ]
    EXEMPTION_EN = [
        r"\bnot required to\b", r"\bis not obligated to\b", r"\bunder no obligation\b",
        r"\bexempt from\b", r"\brelieved of\b", r"\bwithout obligation\b",
        r"\bcan\b", r"\bmay\b", r"\bis permitted to\b", r"\bare permitted to\b",
        r"\bis allowed to\b", r"\bare allowed to\b", r"\bhas the right to\b",
        r"\bhave the right to\b", r"\bentitled to\b", r"\bpermitted to\b"
    ]

    OBLIGATION_FR = [
        r"\bdoit\b", r"\bdoivent\b", r"\best tenu de\b", r"\bsont tenus de\b",
        r"\bs'engage à\b", r"\ba l'obligation de\b", r"\bobligatoirement\b"
    ]
    PROHIBITION_FR = [
        r"\bne doit pas\b", r"\bne doivent pas\b", r"\bne peut pas\b", r"\bne peuvent pas\b",
        r"\binterdit\b", r"\bdéfense de\b", r"\bprohibé\b", r"\ben aucun cas\b"
    ]
    EXEMPTION_FR = [
        r"\bn'est pas tenu de\b", r"\bne sont pas tenus de\b", r"\bdispensé de\b",
        r"\bdispensée de\b", r"\bnon requis\b", r"\baucune obligation\b", r"\bsans obligation\b"
    ]

    OBLIGATION_HI = [
        r"होगा", r"होगी", r"होंगे", r"चाहिए", r"अनिवार्य है", r"बाध्य है", r"आवश्यक है", r"कर्तव्य है",
        r"निर्देश दिया", r"आदेश दिया", r"बाध्य किया", r"लागू होगी"
    ]
    PROHIBITION_HI = [
        r"मनाही है", r"सख्त मनाही", r"वर्जित है", r"प्रतिबंधित है", r"अनुमति नहीं", r"नहीं कर सकते",
        r"नहीं होना चाहिए", r"नहीं.*?चाहिए", r"नहीं दी जानी चाहिए", r"नहीं दिया जाना चाहिए",
        r"अधिकार नहीं", r"निषेध", r"खारिज कर दिया", r"खारिज किया",
        r"अनुमति नहीं दी जा सकती", r"इनकार कर दिया", r"अस्वीकार किया", r"लागू नहीं होगी"
    ]
    EXEMPTION_HI = [
        r"आवश्यकता नहीं", r"बाध्य नहीं", r"छूट है", r"अनिवार्य नहीं", r"सकते हैं", r"अनुमति है",
        r"जमानत दी", r"जमानत प्रदान की", r"राहत दी", r"स्वीकार किया"
    ]

    # --- Exception Markers ---
    EXCEPTION_EN = [
        "unless", "except as provided", "except when", "except in the case",
        "provided that", "subject to", "with the exception of", "saving and excepting",
        "save and except", "save that"
    ]
    EXCEPTION_FR = [
        "sauf si", "sauf en cas", "sauf", "à l'exception de", "à moins que",
        "sous réserve de", "hormis", "excepté", "pourvu que"
    ]
    EXCEPTION_HI = [
        "सिवाय", "जब तक कि", "बशर्ते कि", "को छोड़कर", "यदि नहीं"
    ]

    # --- Temporal & Numerical Regex ---
    TEMPORAL_PATTERNS = [
        r"\b(\d+)\s*(days?|business days?|calendar days?|weeks?|months?|years?|hours?)\b",
        r"\b(\d+)\s*(jours?|jours ouvrables?|semaines?|mois|ans?|années?|heures?)\b"
    ]
    CURRENCY_PATTERNS = [
        r"\$\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*(?:€|EUR|USD|\$|dollars?|euros?)"
    ]
    PERCENT_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*(?:%|pourcent|percent|per cent)"
    ]

    # --- Contract Party Roles ---
    PARTY_PAIRS = [
        # (Pattern A, Pattern B, Label A, Label B)
        (r"\bdisclosing part(?:y|ies)\b", r"\breceiving part(?:y|ies)\b", "Disclosing Party", "Receiving Party"),
        (r"\bdiscloser\b", r"\brecipient\b", "Discloser", "Recipient"),
        (r"\blandlord\b", r"\btenant\b", "Landlord (Bailleur)", "Tenant (Locataire)"),
        (r"\blessor\b", r"\blessee\b", "Lessor (Bailleur)", "Lessee (Preneur)"),
        (r"\blicensor\b", r"\blicensee\b", "Licensor (Concédant)", "Licensee (Licencié)"),
        (r"\bemployer\b", r"\bemployee\b", "Employer (Employeur)", "Employee (Salarié)"),
        (r"\bbuyer\b", r"\bseller\b", "Buyer (Acheteur)", "Seller (Vendeur)"),
        (r"\bpurchaser\b", r"\bvendor\b", "Purchaser (Acquéreur)", "Vendor (Vendeur)"),
        (r"\binsurer\b", r"\binsured\b", "Insurer (Assureur)", "Insured (Assuré)"),
        (r"\bcreditor\b", r"\bdebtor\b", "Creditor (Créancier)", "Debtor (Débiteur)"),
        (r"\blender\b", r"\bborrower\b", "Lender (Prêteur)", "Borrower (Emprunteur)"),
        (r"\bclient\b", r"\b(?:service )?provider\b", "Client", "Service Provider"),
        (r"\bcompany\b", r"\b(?:contractor|consultant)\b", "Company", "Contractor/Consultant"),
        (r"\bindemnifying part(?:y|ies)\b", r"\bindemnified part(?:y|ies)\b", "Indemnifying Party", "Indemnified Party"),
        # French
        (r"\bl['’]assureur\b", r"\bl['’]assur[ée]\b", "L'assureur", "L'assuré"),
        (r"\bassureur\b", r"\bassur[ée]\b", "Assureur", "Assuré"),
        (r"\bbailleur\b", r"\b(?:locataire|preneur)\b", "Bailleur", "Locataire/Preneur"),
        (r"\bc[ée]dant\b", r"\bcessionnaire\b", "Cédant", "Cessionnaire"),
        (r"\bemployeur\b", r"\b(?:employ[ée]|salari[ée])\b", "Employeur", "Employé/Salarié"),
        (r"\bacheteur\b", r"\bvendeur\b", "Acheteur", "Vendeur"),
        (r"\bclient\b", r"\bprestataire\b", "Client", "Prestataire"),
        (r"\bcr[ée]ancier\b", r"\bd[ée]biteur\b", "Créancier", "Débiteur"),
        # Indian Legal Judgments (MILDSum)
        (r"\bpetitioner\b", r"\brespondent\b", "Petitioner", "Respondent"),
        (r"\bappellant\b", r"\brespondent\b", "Appellant", "Respondent"),
        (r"\bयाचिकाकर्ता\b", r"\bप्रतिवादी\b", "याचिकाकर्ता (Petitioner)", "प्रतिवादी (Respondent)"),
    ]

    @classmethod
    def extract_syntactic_roles(cls, text: str, pat_a: str, pat_b: str, label_a: str, label_b: str) -> Dict[str, str]:
        """
        Syntactic Dependency & Semantic Role Resolver:
        Extracts whether entities act as AGENT, PATIENT, or RECIPIENT.
        Correctly parses active voice, passive voice ('by/par'), and directional prepositions ('to/à').
        """
        text_lower = text.lower()

        # 1. Passive voice detection (EN & FR)
        pass_en = r'\b(is|are|was|were|be|been|being)\s+([a-z]+(?:ed|en|t))\b'
        pass_fr = r'\b(est|sont|été|fut|furent|sera|seront|a été|ont été)\s+([a-zàâéèêëîïôûùüç]+(?:é|ée|és|ées|i|ie|is|ies|u|ue|us|ues|t|te))\b'
        is_passive = bool(re.search(pass_en, text_lower) or re.search(pass_fr, text_lower))

        # 2. Main action verbs & modals
        verbs_en = r'\b(return|returned|returning|pay|paid|paying|provide|provided|providing|indemnify|indemnified|deliver|delivered|acquire|acquired|hold|held|disclose|disclosed|agree|undertake|shall|must|will|sue|sued)\b'
        verbs_fr = r'\b(indemniser|indemnisé|aviser|avisé|payer|payé|rembourser|remboursé|fournir|fourni|doit|doivent|est tenu)\b'
        m_verb = re.search(verbs_en, text_lower) or re.search(verbs_fr, text_lower)
        v_pos = m_verb.start() if m_verb else -1

        def determine_role(pat):
            m = re.search(pat, text_lower, flags=re.IGNORECASE)
            if not m:
                return SyntacticRole.UNKNOWN
            start = m.start()
            preceding = text_lower[max(0, start - 30):start]

            # Governed by passive agent preposition 'by' or 'par'
            if re.search(r'\b(by|par)(?:\s+(?:the|l\x27|la|le|les))?\s*$', preceding):
                return SyntacticRole.AGENT

            # Governed by recipient/directional preposition 'to', 'unto', 'à', 'au', 'aux', 'envers', 'pour'
            if re.search(r'\b(to|unto|à|au|aux|envers|pour)(?:\s+(?:the|l\x27|la|le|les))?\s*$', preceding):
                return SyntacticRole.RECIPIENT

            # Governed by source preposition 'from', 'de', 'du', 'des'
            if re.search(r'\b(from|de|du|des)(?:\s+(?:the|l\x27|la|le|les))?\s*$', preceding):
                return SyntacticRole.RECIPIENT

            # Position-based syntactic dependency on verb
            if v_pos != -1:
                if is_passive:
                    return SyntacticRole.PATIENT if start < v_pos else SyntacticRole.RECIPIENT
                else:
                    return SyntacticRole.AGENT if start < v_pos else SyntacticRole.PATIENT

            return SyntacticRole.UNKNOWN

        return {
            label_a: determine_role(pat_a),
            label_b: determine_role(pat_b)
        }

    @classmethod
    def check_party_inversion(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Syntactic Dependency & Semantic Role Guardrail for Contractual Parties:
        Detects Actor-Role Binding Reversals while handling active/passive voice equivalence.
        Returns: (is_violation, explanation, penalty)
        """
        orig_lower = orig.lower()
        simp_lower = simp.lower()

        for pat_a, pat_b, label_a, label_b in cls.PARTY_PAIRS:
            match_a_orig = re.search(pat_a, orig_lower)
            match_b_orig = re.search(pat_b, orig_lower)
            match_a_simp = re.search(pat_a, simp_lower)
            match_b_simp = re.search(pat_b, simp_lower)

            # Case 1: Both parties present in both sentences
            if match_a_orig and match_b_orig and match_a_simp and match_b_simp:
                roles_orig = cls.extract_syntactic_roles(orig, pat_a, pat_b, label_a, label_b)
                roles_simp = cls.extract_syntactic_roles(simp, pat_a, pat_b, label_a, label_b)

                ra_o, rb_o = roles_orig[label_a], roles_orig[label_b]
                ra_s, rb_s = roles_simp[label_a], roles_simp[label_b]

                # If syntactic roles were successfully extracted and distinct:
                if ra_o != SyntacticRole.UNKNOWN and rb_o != SyntacticRole.UNKNOWN and ra_o != rb_o:
                    # True syntactic role reversal
                    if ra_s == rb_o and rb_s == ra_o:
                        return True, f"Party Inversion (Inversion des parties) : Contractual roles of '{label_a}' and '{label_b}' were reversed.", 9.5
                    elif ra_s == ra_o and rb_s == rb_o:
                        # Verified syntactic equivalence (e.g. valid passive voice transformation)
                        continue

                # Positional dependency fallback
                order_orig = match_a_orig.start() < match_b_orig.start()
                order_simp = match_a_simp.start() < match_b_simp.start()
                if order_orig != order_simp:
                    return True, f"Party Inversion (Inversion des parties) : Contractual roles of '{label_a}' and '{label_b}' were reversed.", 9.5

            # Case 2: Obligation/Role substitution (Only one party present in orig, but replaced by opposite party in simp)
            if match_a_orig and not match_b_orig and match_b_simp and not match_a_simp:
                return True, f"Party Substitution : Obligation or role of '{label_a}' was wrongly transferred to '{label_b}'.", 9.5
            if match_b_orig and not match_a_orig and match_a_simp and not match_b_simp:
                return True, f"Party Substitution : Obligation or role of '{label_b}' was wrongly transferred to '{label_a}'.", 9.5

        return False, "", 0.0

    @classmethod
    def check_deontic_polarity(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Detects contradiction in legal obligations, prohibitions, and exemptions.
        Returns: (is_violation, explanation, penalty)
        """
        orig_lower = orig.lower()
        simp_lower = simp.lower()

        # Check English
        has_ob_en1 = any(re.search(p, orig_lower) for p in cls.OBLIGATION_EN)
        has_proh_en1 = any(re.search(p, orig_lower) for p in cls.PROHIBITION_EN)
        has_exemp_en1 = any(re.search(p, orig_lower) for p in cls.EXEMPTION_EN)

        has_ob_en2 = any(re.search(p, simp_lower) for p in cls.OBLIGATION_EN)
        has_proh_en2 = any(re.search(p, simp_lower) for p in cls.PROHIBITION_EN)
        has_exemp_en2 = any(re.search(p, simp_lower) for p in cls.EXEMPTION_EN)

        # Check French
        has_ob_fr1 = any(re.search(p, orig_lower) for p in cls.OBLIGATION_FR)
        has_proh_fr1 = any(re.search(p, orig_lower) for p in cls.PROHIBITION_FR)
        has_exemp_fr1 = any(re.search(p, orig_lower) for p in cls.EXEMPTION_FR)

        has_ob_fr2 = any(re.search(p, simp_lower) for p in cls.OBLIGATION_FR)
        has_proh_fr2 = any(re.search(p, simp_lower) for p in cls.PROHIBITION_FR)
        has_exemp_fr2 = any(re.search(p, simp_lower) for p in cls.EXEMPTION_FR)

        # Check Hindi
        has_ob_hi1 = any(re.search(p, orig_lower) for p in cls.OBLIGATION_HI)
        has_proh_hi1 = any(re.search(p, orig_lower) for p in cls.PROHIBITION_HI)
        has_exemp_hi1 = any(re.search(p, orig_lower) for p in cls.EXEMPTION_HI)

        has_ob_hi2 = any(re.search(p, simp_lower) for p in cls.OBLIGATION_HI)
        has_proh_hi2 = any(re.search(p, simp_lower) for p in cls.PROHIBITION_HI)
        has_exemp_hi2 = any(re.search(p, simp_lower) for p in cls.EXEMPTION_HI)

        is_ob1 = has_ob_en1 or has_ob_fr1 or has_ob_hi1
        is_proh1 = has_proh_en1 or has_proh_fr1 or has_proh_hi1
        is_exemp1 = has_exemp_en1 or has_exemp_fr1 or has_exemp_hi1

        is_ob2 = has_ob_en2 or has_ob_fr2 or has_ob_hi2
        is_proh2 = has_proh_en2 or has_proh_fr2 or has_proh_hi2
        is_exemp2 = has_exemp_en2 or has_exemp_fr2 or has_exemp_hi2

        # 1. Direct flip: Positive Obligation -> Prohibition
        if (is_ob1 and not is_proh1) and is_proh2:
            return True, "Obligation Reversal : A contractual obligation became a prohibition.", 9.0

        # 2. Direct flip: Prohibition -> Positive Obligation or Permission
        if is_proh1 and (is_ob2 or is_exemp2) and not is_proh2:
            return True, "Prohibition Reversal : A formal prohibition became an obligation or permission.", 9.0

        # 3. Obligation -> Exemption
        if (is_ob1 and not is_exemp1) and is_exemp2:
            return True, "Obligation Cancellation (Annulation d'obligation) : The contractual obligation was removed or made optional.", 8.5

        # 4. Coverage vs Exclusion
        excl_orig = any(w in orig_lower for w in ["exclut", "exclusion", "excludes", "does not cover", "ne couvre pas"])
        excl_simp = any(w in simp_lower for w in ["exclut", "exclusion", "excludes", "does not cover", "ne couvre pas"])
        cov_orig = any(w in orig_lower for w in ["couvre", "garantit", "covers", "includes", "assures"])
        cov_simp = any(w in simp_lower for w in ["couvre", "garantit", "covers", "includes", "assures"])

        if excl_orig and (cov_simp and not excl_simp):
            return True, "Coverage Inversion : An exclusion clause was turned into positive coverage.", 9.0
        if (cov_orig and not excl_orig) and excl_simp:
            return True, "Coverage Inversion : A contractual coverage was turned into an exclusion.", 9.0

        # If both are established positive obligations with no prohibitions, polarity is preserved
        if (is_ob1 and not is_proh1 and not is_exemp1) and (is_ob2 and not is_proh2 and not is_exemp2):
            return False, "", 0.0

        # If both are established prohibitions, polarity is preserved
        if is_proh1 and is_proh2:
            return False, "", 0.0

        # 5. Generic Negation mismatch on key predicates (excluding adverbial prepositions like without/sans)
        neg_en = {'not', 'never', 'neither', 'nor', 'cannot', "can't", "won't", "prohibited", "forbidden", "banned"}
        neg_fr = {'ne', 'pas', 'aucun', 'aucune', 'nul', 'nulle', 'jamais', 'interdit', 'interdite', 'interdits', 'interdites', 'prohibé', 'défendu'}
        neg_hi = {'नहीं', 'मत', 'ना', 'वर्जित', 'मनाही', 'निषेध', 'प्रतिबंधित'}
        w1 = set(orig_lower.split()) | set(re.findall(r'\b[a-zA-Z\xc0-\xff]+\b', orig_lower))
        w2 = set(simp_lower.split()) | set(re.findall(r'\b[a-zA-Z\xc0-\xff]+\b', simp_lower))
        has_neg1 = bool(w1 & (neg_en | neg_fr | neg_hi))
        has_neg2 = bool(w2 & (neg_en | neg_fr | neg_hi))

        if has_neg1 != has_neg2 and ((is_ob1 and not is_proh1) or (is_ob2 and not is_proh2) or excl_orig or excl_simp):
            return True, "Polarity Inversion : Conflicting negation detected in core legal obligation.", 8.5

        return False, "", 0.0

    @classmethod
    def check_critical_exceptions(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Detects if an essential legal exception or restriction was dropped:
        e.g., 'unless', 'except', 'provided that', 'subject to', 'sauf si'.
        Returns: (is_violation, explanation, penalty)
        """
        orig_lower = orig.lower()
        simp_lower = simp.lower()

        all_exceptions = cls.EXCEPTION_EN + cls.EXCEPTION_FR + cls.EXCEPTION_HI
        exc_found_orig = [m for m in all_exceptions if m in orig_lower]
        exc_found_simp = [m for m in all_exceptions if m in simp_lower]

        if exc_found_orig and not exc_found_simp:
            markers_str = ", ".join(f"'{m}'" for m in exc_found_orig[:2])
            return True, f"Critical Exception Omitted ({markers_str}). This omission improperly expands the scope of the clause.", 3.5

        return False, "", 0.0

    @classmethod
    def extract_constraints(cls, text: str) -> Dict[str, List[str]]:
        """
        Extracts numerical, monetary, and temporal constraints from text.
        """
        text_lower = text.lower()
        temporal = []
        for pat in cls.TEMPORAL_PATTERNS:
            for match in re.finditer(pat, text_lower):
                temporal.append(f"{match.group(1)} {match.group(2)}")

        currencies = []
        for pat in cls.CURRENCY_PATTERNS:
            for match in re.finditer(pat, text):
                currencies.append(match.group(0).strip())

        percentages = []
        for pat in cls.PERCENT_PATTERNS:
            for match in re.finditer(pat, text):
                percentages.append(match.group(0).strip())

        # Standalone numbers (e.g. counts, thresholds, dates) - normalize decimal commas to dots
        raw_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
        numbers = [n.replace(',', '.') for n in raw_numbers]

        return {
            "temporal": temporal,
            "currencies": currencies,
            "percentages": percentages,
            "numbers": numbers,
        }

    @classmethod
    def check_constraints_match(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Detects numerical, monetary, or temporal constraint alterations or omissions.
        Returns: (is_violation, explanation, penalty)
        """
        c_orig = cls.extract_constraints(orig)
        c_simp = cls.extract_constraints(simp)

        violations = []
        penalty = 0.0

        # Check temporal durations (e.g., 30 days vs 60 days, or dropped duration)
        if c_orig["temporal"] and not c_simp["temporal"]:
            durations = ", ".join(c_orig["temporal"])
            violations.append(f"Contractual deadline omitted : '{durations}' missing from simplification.")
            penalty += 3.0
        elif c_orig["temporal"] and c_simp["temporal"]:
            # Check numbers in durations
            nums_orig = [re.search(r'\d+', t).group(0) for t in c_orig["temporal"] if re.search(r'\d+', t)]
            nums_simp = [re.search(r'\d+', t).group(0) for t in c_simp["temporal"] if re.search(r'\d+', t)]
            if set(nums_orig) != set(nums_simp):
                violations.append(f"Modification de délai / Deadline altered : '{', '.join(c_orig['temporal'])}' altered to '{', '.join(c_simp['temporal'])}'.")
                penalty += 4.5

        # Check currencies / monetary amounts
        if c_orig["currencies"] and not c_simp["currencies"]:
            violations.append(f"Financial amount omitted : '{', '.join(c_orig['currencies'])}' missing.")
            penalty += 3.5
        elif c_orig["currencies"] and c_simp["currencies"]:
            # Clean amounts to digits
            clean_orig = [re.sub(r'[^\d]', '', c) for c in c_orig["currencies"]]
            clean_simp = [re.sub(r'[^\d]', '', c) for c in c_simp["currencies"]]
            if set(clean_orig) != set(clean_simp):
                violations.append(f"Financial amount altered (altéré) : '{', '.join(c_orig['currencies'])}' vs '{', '.join(c_simp['currencies'])}'.")
                penalty += 5.0

        # Check percentages
        if c_orig["percentages"] and not c_simp["percentages"]:
            violations.append(f"Rate or percentage omitted : '{', '.join(c_orig['percentages'])}'.")
            penalty += 2.5
        elif c_orig["percentages"] and c_simp["percentages"]:
            nums_orig = [re.sub(r'[^\d.]', '', p) for p in c_orig["percentages"]]
            nums_simp = [re.sub(r'[^\d.]', '', p) for p in c_simp["percentages"]]
            if set(nums_orig) != set(nums_simp):
                violations.append(f"Percentage altered (altéré) : '{', '.join(c_orig['percentages'])}' vs '{', '.join(c_simp['percentages'])}'.")
                penalty += 4.0

        # Check standalone numbers / quantities / dates (e.g. 1889 vs 1989, 330 vs 130)
        nums_orig = set(c_orig.get("numbers", []))
        nums_simp = set(c_simp.get("numbers", []))
        diff_nums = (nums_orig - nums_simp) | (nums_simp - nums_orig)
        if diff_nums and len(nums_orig) > 0 and not c_orig["temporal"] and not c_orig["currencies"] and not c_orig["percentages"]:
            violations.append(f"Numerical parameters or dates altered (altéré) : '{', '.join(diff_nums)}'.")
            penalty += 4.5

        if violations:
            return True, " | ".join(violations), min(penalty, 6.0)

        return False, "", 0.0

    @classmethod
    def check_causal_inversion(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Detects causal inversion (Cause <-> Effect reversal):
        e.g., 'A was cancelled because of B' -> 'B happened because of A'.
        """
        orig_lower = orig.lower()
        simp_lower = simp.lower()

        causal_markers_en = [r"\bbecause\b", r"\bdue to\b", r"\bas a result of\b", r"\bowing to\b"]
        causal_markers_fr = [r"\bparce que\b", r"\ben raison de\b", r"\bà cause de\b", r"\bcar\b"]

        pat_causal = "|".join(causal_markers_en + causal_markers_fr)

        # Check if both have a causal marker
        m_orig = re.search(pat_causal, orig_lower)
        m_simp = re.search(pat_causal, simp_lower)

        if m_orig and m_simp:
            # Split into Clause 1 (effect/cause) and Clause 2 (cause/effect)
            before_orig = set(re.findall(r'\b\w{4,}\b', orig_lower[:m_orig.start()]))
            after_orig = set(re.findall(r'\b\w{4,}\b', orig_lower[m_orig.end():]))

            before_simp = set(re.findall(r'\b\w{4,}\b', simp_lower[:m_simp.start()]))
            after_simp = set(re.findall(r'\b\w{4,}\b', simp_lower[m_simp.end():]))

            # If words before marker in orig overlap heavily with words AFTER marker in simp,
            # and words after marker in orig overlap heavily with words BEFORE marker in simp:
            overlap_c1 = len(before_orig & after_simp)
            overlap_c2 = len(after_orig & before_simp)
            direct_c1 = len(before_orig & before_simp)
            direct_c2 = len(after_orig & after_simp)

            if overlap_c1 >= 2 and overlap_c2 >= 2 and (overlap_c1 + overlap_c2) > (direct_c1 + direct_c2):
                return True, "Causal Inversion (Inversion de causalité - Cause ↔ Effect) : The cause and consequence were inverted.", 9.0

        return False, "", 0.0

    @classmethod
    def extract_named_entities(cls, text: str) -> List[str]:
        """
        Extracts proper nouns and named entities (e.g. Apple, NeXT, Paris, NASA).
        """
        # Words with capital letters not at the strict start of a sentence
        tokens = text.split()
        entities = []
        for i, tok in enumerate(tokens):
            clean = tok.strip(".,;:?!'\"()[]{}")
            if not clean:
                continue
            # If capitalized and either not first word or title-cased acronym/special entity
            if clean[0].isupper() and (i > 0 or clean.isupper()):
                # Exclude common sentence starters and common modals
                common_stops = {
                    "the", "this", "that", "these", "those", "each", "every", "all", "none",
                    "neither", "either", "shall", "must", "with", "from", "into", "subdivision",
                    "section", "article", "clause", "agreement", "contract", "party", "parties",
                    "partie", "disclosing", "receiving"
                }
                if clean.lower() not in common_stops:
                    entities.append(clean)
        return list(dict.fromkeys(entities))

    @classmethod
    def check_entity_inversion(cls, orig: str, simp: str) -> Tuple[bool, str, float]:
        """
        Detects general Subject-Object / Actor inversion for proper entities:
        e.g., 'Apple acquired NeXT' -> 'NeXT acquired Apple'.
        """
        ent_orig = cls.extract_named_entities(orig)
        ent_simp = cls.extract_named_entities(simp)

        # Look for two entities that exist in both sentences
        common_entities = [e for e in ent_orig if any(e.lower() == s.lower() for s in ent_simp)]
        if len(common_entities) >= 2:
            e1 = common_entities[0]
            e2 = common_entities[1]
            pat_1 = r'\b' + re.escape(e1) + r'\b'
            pat_2 = r'\b' + re.escape(e2) + r'\b'

            # 1. Syntactic Role Analysis (Handles active/passive voice transformation without false positives)
            roles_orig = cls.extract_syntactic_roles(orig, pat_1, pat_2, e1, e2)
            roles_simp = cls.extract_syntactic_roles(simp, pat_1, pat_2, e1, e2)

            r1_o, r2_o = roles_orig[e1], roles_orig[e2]
            r1_s, r2_s = roles_simp[e1], roles_simp[e2]

            if r1_o != SyntacticRole.UNKNOWN and r2_o != SyntacticRole.UNKNOWN and r1_o != r2_o:
                if r1_s == r2_o and r2_s == r1_o:
                    return True, f"Actor / Entity Inversion (Inversion d'acteurs - Agent ↔ Patient) : The roles of '{e1}' and '{e2}' were inverted.", 9.0
                elif r1_s == r1_o and r2_s == r2_o:
                    # Syntactically verified preservation (e.g. passive transformation 'was acquired by')
                    return False, "", 0.0

            # 2. Positional dependency fallback
            pos_e1_orig = orig.lower().find(e1.lower())
            pos_e2_orig = orig.lower().find(e2.lower())

            pos_e1_simp = simp.lower().find(e1.lower())
            pos_e2_simp = simp.lower().find(e2.lower())

            if pos_e1_orig != -1 and pos_e2_orig != -1 and pos_e1_simp != -1 and pos_e2_simp != -1:
                order_orig = pos_e1_orig < pos_e2_orig
                order_simp = pos_e1_simp < pos_e2_simp

                if order_orig != order_simp:
                    return True, f"Actor / Entity Inversion (Inversion d'acteurs - Agent ↔ Patient) : The roles of '{e1}' and '{e2}' were inverted.", 9.0

        return False, "", 0.0

    @classmethod
    def detect_domain(cls, text: str) -> str:
        """
        Auto-detects whether the text is Legal/Contractual or General Context.
        """
        text_lower = text.lower()
        legal_markers = [
            "shall", "pursuant", "hereunder", "indemnif", "liability", "breach", "covenant",
            "lessee", "lessor", "licensor", "licensee", "disclosing party", "receiving party",
            "prohibit", "forbidden", "interdit", "confidential", "tenant", "landlord",
            "avenant", "assur", "sinistre", "indemnité", "subrogation", "résiliation",
            "bailleur", "preneur", "contractuel", "exclut la garantie", "dispositions", "code civil",
            # Hindi Legal & Contractual Markers
            "मनाही", "वर्जित", "प्रतिबंधित", "किराया", "अनुबंध", "शर्त", "दायित्व", "बाध्य",
            "अधिकार", "बीमा", "हर्जाना", "उल्लंघन", "गोपनीय", "कर्मचारी", "किराएदार",
            # Indian Legal Judgments & Statutes (MILDSum)
            "court", "high court", "supreme court", "judgment", "ruling", "petitioner", "respondent",
            "accused", "bail", "ndps", "act", "section", "decree", "order", "quashed", "reimbursement",
            "अदालत", "कोर्ट", "हाईकोर्ट", "सुप्रीम कोर्ट", "फैसले", "फैसला", "निर्णय", "याचिका",
            "याचिकाकर्ता", "प्रतिवादी", "आरोपी", "अभियुक्त", "जमानत", "धारा", "अधिनियम", "डिक्री",
            "आदेश", "रद्द", "प्रतिपूर्ति", "दोषी", "दोषमुक्त", "खंडपीठ", "पूर्वाग्रह", "प्रावधान"
        ]
        if any(m in text_lower for m in legal_markers):
            return "legal"
        return "general"

    @classmethod
    def verify_all_guardrails(cls, orig: str, simp: str, domain: str = "auto") -> Dict[str, Any]:
        """
        Runs comprehensive guardrails supporting both General Context and Legal Contracts.
        """
        if domain == "auto":
            domain = cls.detect_domain(orig + " " + simp)

        c_orig = cls.extract_constraints(orig)
        c_simp = cls.extract_constraints(simp)

        alerts = []
        is_severe_contradiction = False

        # --- UNIVERSAL GUARDRAILS (Run on all domains) ---
        # 1. Causal Directionality (Cause <-> Effect)
        v_causal, msg_causal, pen_causal = cls.check_causal_inversion(orig, simp)
        if v_causal:
            alerts.append({"type": "causal_inversion", "level": "danger", "message": msg_causal, "penalty": pen_causal})
            is_severe_contradiction = True

        # 2. General Named Entity Inversion (Agent <-> Patient)
        v_ent, msg_ent, pen_ent = cls.check_entity_inversion(orig, simp)
        if v_ent:
            alerts.append({"type": "entity_inversion", "level": "danger", "message": msg_ent, "penalty": pen_ent})
            is_severe_contradiction = True

        # 3. Numerical, Temporal & Monetary Precision
        v_const, msg_const, pen_const = cls.check_constraints_match(orig, simp)
        if v_const:
            alerts.append({"type": "constraint_mismatch", "level": "warning", "message": msg_const, "penalty": pen_const})

        # --- DOMAIN-SPECIFIC GUARDRAILS ---
        pen_deontic, pen_party, pen_exc = 0.0, 0.0, 0.0
        if domain == "legal":
            # Legal Layer 1: Deontic Modals & Polarity
            v_deontic, msg_deontic, pen_deontic = cls.check_deontic_polarity(orig, simp)
            if v_deontic:
                alerts.append({"type": "deontic_polarity", "level": "danger", "message": msg_deontic, "penalty": pen_deontic})
                is_severe_contradiction = True

            # Legal Layer 2: Actor-Role & Party Inversion
            v_party, msg_party, pen_party = cls.check_party_inversion(orig, simp)
            if v_party:
                alerts.append({"type": "party_inversion", "level": "danger", "message": msg_party, "penalty": pen_party})
                is_severe_contradiction = True

            # Legal Layer 3: Exceptions & Conditions
            v_exc, msg_exc, pen_exc = cls.check_critical_exceptions(orig, simp)
            if v_exc:
                alerts.append({"type": "critical_exception", "level": "warning", "message": msg_exc, "penalty": pen_exc})
        else:
            # General Polarity / Negation mismatch (including inherent negative/prohibitive predicates)
            neg_en = {'not', 'never', 'neither', 'nor', 'cannot', "can't", "won't", "prohibited", "forbidden", "banned"}
            neg_fr = {'ne', 'pas', 'aucun', 'aucune', 'nul', 'nulle', 'jamais', 'interdit', 'interdite', 'interdits', 'interdites', 'prohibé', 'défendu'}
            w1 = set(re.findall(r'\b[a-zA-Z\xc0-\xff]+\b', orig.lower()))
            w2 = set(re.findall(r'\b[a-zA-Z\xc0-\xff]+\b', simp.lower()))
            has_neg1 = bool(w1 & (neg_en | neg_fr))
            has_neg2 = bool(w2 & (neg_en | neg_fr))
            if has_neg1 != has_neg2:
                alerts.append({
                    "type": "general_polarity",
                    "level": "danger",
                    "message": "General polarity inversion (affirmation vs negation - Inversion de polarité générale).",
                    "penalty": 8.5
                })
                is_severe_contradiction = True

        total_penalty = pen_causal + pen_ent + pen_const + pen_deontic + pen_party + pen_exc

        return {
            "domain": domain,
            "passed": len(alerts) == 0,
            "alerts": alerts,
            "total_penalty": total_penalty,
            "is_severe_contradiction": is_severe_contradiction,
            "entities_orig": c_orig,
            "entities_simp": c_simp,
        }
