import io
import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class DocumentContextMemory:
    """
    Tracks defined contract aliases (e.g. Acme Holdings LLC ("Vendor")),
    section headings, and active entity antecedents across sequential clauses
    to resolve pronouns (he, she, it, they, ils, elles) and defined terms
    before per-clause NLI evaluation.
    """

    # Parenthetical alias definitions like: Acme Corp ("Vendor") or Acme LLC (hereinafter "Supplier")
    _ALIAS_DEF_RE = re.compile(
        r"\b([A-Z][A-Za-z0-9&\.\-]*(?:\s+(?:of|and|&|[A-Z][A-Za-z0-9&\.\-]*)){0,5})\s*"
        r"\(\s*(?:the\s+|hereinafter\s+(?:referred\s+to\s+as\s+|called\s+)?)?[\"\'“”]([A-Za-z][A-Za-z\s]{1,25})[\"\'“”]\s*\)"
    )

    # Active subject extraction patterns
    _ORG_ENTITY_RE = re.compile(
        r"\b([A-Z][A-Za-z0-9&\.\-]*(?:\s+(?:of|and|&|[A-Z][A-Za-z0-9&\.\-]*)){0,4}\s+"
        r"(?:Pvt\.?\s+Ltd\.?|Private\s+Limited|LLC|Ltd\.?|Inc\.?|Corp\.?|Corporation|Limited|LLP|PLC|GmbH|SARL|Holdings|Bank|Company))\b"
    )
    _ROLE_ENTITY_RE = re.compile(
        r"\b(?:The\s+|Le\s+|La\s+|L\')?(Receiving Party|Disclosing Party|Tenant|Landlord|Contractor|Supplier|Vendor|Insurer|Insured|Employee|Employer|Lessor|Lessee|Licensor|Licensee|Assureur|Assuré|Locataire)\b",
        re.IGNORECASE,
    )
    _PERSON_MALE_RE = re.compile(r"\b(?:Mr\.?|Monsieur|M\.|Shri)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
    _PERSON_FEMALE_RE = re.compile(r"\b(?:Mrs\.?|Ms\.?|Miss|Mme|Madame|Mlle|Smt\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

    _SECTION_HEADER_RE = re.compile(r"^\s*(?:Section|Clause|Article|Art\.?)\s+(\d+(?:\.\d+)*)\s*[:\-\.]?\s*", re.IGNORECASE)

    # Dummy/pleonastic "it" check (do not replace "It is prohibited...")
    _DUMMY_IT_RE = re.compile(r"\b(?:it|ce|il)\s+(?:is|was|shall\s+be|will\s+be|may\s+be|can\s+be|must\s+be|appears|seems|est)\b", re.IGNORECASE)

    def __init__(self, shared_aliases: Optional[Dict[str, str]] = None):
        self.aliases: Dict[str, str] = dict(shared_aliases or {})
        self.section_summaries: Dict[str, str] = {}
        self.last_org: Optional[str] = None
        self.last_role: Optional[str] = None
        self.last_male: Optional[str] = None
        self.last_female: Optional[str] = None
        self.last_plural: Optional[str] = None

    def observe_clause(self, clause: str) -> None:
        """Extracts new definitions, headings, and candidate entities from a clause."""
        if not clause:
            return

        # 1. Parenthetical legal definitions
        for m in self._ALIAS_DEF_RE.finditer(clause):
            full_name = m.group(1).strip()
            alias = m.group(2).strip()
            if full_name and alias and len(alias.split()) <= 4:
                self.aliases[alias.lower()] = full_name
                self.last_org = full_name

        # 2. Section header memory
        sec_m = self._SECTION_HEADER_RE.match(clause)
        if sec_m:
            sec_key = f"section {sec_m.group(1)}"
            body = clause[sec_m.end():].strip()
            if body:
                self.section_summaries[sec_key] = body[:120]

        # 3. Organization entities
        org_m = self._ORG_ENTITY_RE.search(clause)
        if org_m:
            self.last_org = org_m.group(1).strip().rstrip(".")
            # Organizations are often referred to via both "It" and corporate plural "They"
            self.last_plural = self.last_org

        # 4. Standard contractual roles (resolve through aliases if previously defined)
        role_m = self._ROLE_ENTITY_RE.search(clause)
        if role_m:
            raw_role = role_m.group(0).strip()
            self.last_role = self.aliases.get(raw_role.lower(), raw_role)
            if not self.last_plural and any(w in clause.lower() for w in ["parties", "employees", "tenants", "contractors"]):
                self.last_plural = self.last_role

        # 5. Personal entities (gender-tagged for pronoun disambiguation)
        male_m = self._PERSON_MALE_RE.search(clause)
        if male_m:
            self.last_male = male_m.group(0).strip()

        female_m = self._PERSON_FEMALE_RE.search(clause)
        if female_m:
            self.last_female = female_m.group(0).strip()

        if male_m and female_m:
            self.last_plural = f"{male_m.group(0).strip()} and {female_m.group(0).strip()}"

    def resolve_clause(self, clause: str) -> Tuple[str, List[str]]:
        """
        Resolves defined-term aliases and third-person pronouns in `clause`
        using accumulated document memory. Also strips inline definition
        parentheticals (`("Vendor")`) cleanly for cleaner semantic comparison.
        Returns (resolved_text, list_of_resolution_notes).
        """
        resolved = clause
        notes: List[str] = []

        # Strip inline definition wrapper like `Acme Holdings LLC ("Vendor")` -> `Acme Holdings LLC`
        # after recording the alias so Doc A and Doc B compare apples to apples
        if self._ALIAS_DEF_RE.search(resolved):
            self.observe_clause(resolved)
            resolved = self._ALIAS_DEF_RE.sub(lambda m: m.group(1).strip(), resolved)

        # 1. Expand defined aliases (e.g. "the Vendor" or "Vendor")
        for alias_lower, full_name in sorted(self.aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
            if full_name.lower() in resolved.lower():
                continue
            pat = re.compile(rf"\b(?:the\s+)?{re.escape(alias_lower)}\b", re.IGNORECASE)
            if pat.search(resolved):
                resolved = pat.sub(full_name, resolved)
                notes.append(f"Defined term '{alias_lower.title()}' ➔ '{full_name}'")

        # 2. Resolve pronouns if an antecedent is available
        active_singular = self.last_org or self.last_role
        active_plural = self.last_plural or self.last_org or self.last_role

        # Male pronouns: He / His / Him
        if self.last_male and re.search(r"\b(?:he|him)\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\b(?:he|him)\b", self.last_male, resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'He/Him' ➔ '{self.last_male}'")
        if self.last_male and re.search(r"\bhis\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\bhis\b", f"{self.last_male}'s", resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'His' ➔ '{self.last_male}'s'")

        # Female pronouns: She / Her
        if self.last_female and re.search(r"\b(?:she)\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\bshe\b", self.last_female, resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'She' ➔ '{self.last_female}'")
        if self.last_female and re.search(r"\bher\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\bher\b", f"{self.last_female}'s", resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'Her' ➔ '{self.last_female}'s'")

        # Corporate/plural pronouns: They / Them / Their
        if active_plural and re.search(r"\b(?:they|them)\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\b(?:they|them)\b", active_plural, resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'They/Them' ➔ '{active_plural}'")
        if active_plural and re.search(r"\btheir\b", resolved, re.IGNORECASE):
            resolved = re.sub(r"\btheir\b", f"{active_plural}'s", resolved, count=1, flags=re.IGNORECASE)
            notes.append(f"Pronoun 'Their' ➔ '{active_plural}'s'")

        # Singular non-dummy entity pronoun: It / Its
        if active_singular and not self._DUMMY_IT_RE.search(resolved):
            if re.search(r"\bit\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\bit\b", active_singular, resolved, count=1, flags=re.IGNORECASE)
                notes.append(f"Pronoun 'It' ➔ '{active_singular}'")
            elif re.search(r"\bits\b", resolved, re.IGNORECASE):
                resolved = re.sub(r"\bits\b", f"{active_singular}'s", resolved, count=1, flags=re.IGNORECASE)
                notes.append(f"Pronoun 'Its' ➔ '{active_singular}'s'")

        # Generic back-references ("aforementioned party", "said entity", "ladite partie")
        if active_singular and re.search(
            r"\b(?:the\s+(?:aforementioned|said|above-named)\s+(?:party|company|entity)|ladite\s+partie)\b",
            resolved,
            re.IGNORECASE,
        ):
            resolved = re.sub(
                r"\b(?:the\s+(?:aforementioned|said|above-named)\s+(?:party|company|entity)|ladite\s+partie)\b",
                active_singular,
                resolved,
                flags=re.IGNORECASE,
            )
            notes.append(f"Reference resolved ➔ '{active_singular}'")

        # Update memory after observing original clause context
        self.observe_clause(clause)
        return resolved, notes


class DocumentEvaluator:
    """
    Whole-Document Legal & Plain-Language Evaluator.

    1. Splits multi-paragraph / multi-sentence agreements into atomic clauses
       without breaking abbreviations (Inc., Ltd., Mr., i.e., Section 3.1).
    2. Applies sequential DocumentContextMemory to resolve defined terms
       and pronouns (he, she, it, they, Vendor) across clauses.
    3. Semantically aligns clauses between Original and Simplification
       using shared JUDGEBERT sentence embeddings (handling 1-to-1, merged,
       split, omitted, and fabricated clauses).
    4. Scores each clause via engine.evaluate_pair(...) and aggregates an
       auditable whole-document preservation scorecard.
    """

    # Abbreviations that must never trigger a sentence split
    _ABBREV_PAT = re.compile(
        r"\b(?:Inc|Ltd|Pvt|Corp|Co|Mr|Mrs|Ms|Dr|Prof|Rs|No|Sec|Art|Clause|Fig|St|Ave|Rd|M|Mme|Mlle|i\.e|e\.g|vs|etc|U\.S|U\.K)\.$",
        re.IGNORECASE,
    )

    @classmethod
    def split_clauses(cls, text: str) -> List[str]:
        """
        Splits a document cleanly into individual sentences/clauses while protecting
        abbreviations, currency decimals, numbered prefixes, and section numbers.
        """
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in re.split(r"\r?\n+", text.strip()) if p.strip()]
        clauses: List[str] = []

        for para in paragraphs:
            # Split at sentence boundaries [.!?।] followed by whitespace
            # Avoid splitting after decimals (3.5), section codes (3.1), or single digit list markers (1.)
            raw_tokens = re.split(r"(?<=[.!?।])\s+(?=[A-Z\u0900-\u097F\"\'“0-9])", para)
            buf = ""
            for chunk in raw_tokens:
                chunk = chunk.strip()
                if not chunk:
                    continue
                if buf:
                    buf = f"{buf} {chunk}"
                else:
                    buf = chunk

                # Check if buf ends with a protected abbreviation or list index like "1." or "a."
                last_word = buf.split()[-1] if buf.split() else ""
                is_protected = (
                    bool(cls._ABBREV_PAT.search(last_word))
                    or bool(re.match(r"^(?:\d{1,2}|[A-Za-z])\.$", last_word))
                    or buf.lower().endswith(("section.", "clause.", "article."))
                )
                if not is_protected:
                    clauses.append(buf)
                    buf = ""
            if buf:
                clauses.append(buf)

        # Clean empty or punctuation-only fragments
        cleaned = [c.strip() for c in clauses if len(c.strip()) > 2 and re.search(r"[\w\u0900-\u097F]", c)]
        return cleaned

    @classmethod
    def is_multi_clause_document(cls, orig: str, simp: str) -> bool:
        """Determines whether inputs represent a multi-clause document (>= 2 clauses)."""
        c_o = cls.split_clauses(orig)
        c_s = cls.split_clauses(simp)
        return len(c_o) >= 2 or len(c_s) >= 2

    @staticmethod
    def _strip_polarity_for_alignment(text: str) -> str:
        """
        Temporarily neutralizes deontic modals and negations solely for topic alignment,
        so contrastive embeddings match a negated/flipped clause back to its true source clause.
        """
        t = text.lower()
        t = re.sub(
            r"\b(?:is not required to|are not required to|is not obligated to|shall not|must not|may not|cannot|not to|no longer|never|not|ne\b|pas\b|n\'est pas tenu de)\b",
            " ",
            t,
        )
        t = re.sub(r"\b(?:shall|must|is required to|agrees to|undertakes to|doit|doivent)\b", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def _content_token_similarity(a: str, b: str) -> float:
        """Normalized content-word overlap score ignoring grammar stopwords & negations."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "at",
            "by", "for", "with", "and", "or", "shall", "must", "will", "not", "required", "has", "have",
            "if", "when", "after", "all", "per", "more", "than", "any", "le", "la", "les", "de", "des",
        }
        toks_a = {w.strip(".,;:!?\"'()") for w in a.lower().split()} - stop - {""}
        toks_b = {w.strip(".,;:!?\"'()") for w in b.lower().split()} - stop - {""}
        if not toks_a or not toks_b:
            return 0.0
        inter = len(toks_a & toks_b)
        overlap_min = inter / min(len(toks_a), len(toks_b))
        jaccard = inter / len(toks_a | toks_b)
        return 0.60 * overlap_min + 0.40 * jaccard

    @classmethod
    def _align_clauses(
        cls,
        orig_resolved: List[str],
        simp_resolved: List[str],
        engine,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Aligns original document clauses with simplified document clauses
        using polarity-neutralized multilingual embeddings + content token overlap
        + positional continuity prior (handling contradictions and omissions cleanly).
        Returns (aligned_pairs, unaligned_simp_indices).
        """
        if not orig_resolved:
            return [], list(range(len(simp_resolved)))
        if not simp_resolved:
            return [{"orig_idxs": [i], "simp_idxs": []} for i in range(len(orig_resolved))], []

        orig_neutral = [cls._strip_polarity_for_alignment(t) for t in orig_resolved]
        simp_neutral = [cls._strip_polarity_for_alignment(t) for t in simp_resolved]
        n_o = len(orig_resolved)
        n_s = len(simp_resolved)

        lex_matrix = np.zeros((n_o, n_s), dtype=float)
        for i in range(n_o):
            for j in range(n_s):
                lex_matrix[i, j] = cls._content_token_similarity(orig_neutral[i], simp_neutral[j])

        if engine.multilingual_model is None:
            sim_matrix = lex_matrix.copy()
        else:
            embs = engine.multilingual_model.encode(orig_neutral + simp_neutral, normalize_embeddings=True)
            dense_matrix = np.matmul(embs[:n_o], embs[n_o:].T)
            # Blend dense cross-lingual topic match with polarity-free lexical overlap
            sim_matrix = np.maximum(dense_matrix, 0.55 * dense_matrix + 0.45 * lex_matrix)

        # Combine semantic topic similarity with gentle monotonic positional prior
        scored_matrix = np.zeros_like(sim_matrix)
        for i in range(n_o):
            pos_i = i / max(n_o - 1, 1)
            for j in range(n_s):
                pos_j = j / max(n_s - 1, 1)
                pos_bonus = 0.08 * (1.0 - abs(pos_i - pos_j))
                scored_matrix[i, j] = float(sim_matrix[i, j]) + pos_bonus

        # Step 1: Assign each simplified clause to its best matching original clause
        simp_to_orig: Dict[int, int] = {}
        for j in range(n_s):
            best_i = int(np.argmax(scored_matrix[:, j]))
            if float(sim_matrix[best_i, j]) >= 0.35:
                simp_to_orig[j] = best_i

        # Build initial per-orig mapping
        orig_to_simps: Dict[int, List[int]] = {i: [] for i in range(n_o)}
        for j, i in simp_to_orig.items():
            orig_to_simps[i].append(j)

        # Step 2: Check if an unmapped original clause matches a simplified sentence cleanly (merged clauses)
        # Wait: if an orig clause got 0 simps, check if it has strong (> 0.55) similarity to an adjacent mapped simp,
        # OR leave as omitted if sim < 0.42 (true omission!)
        aligned_groups: List[Dict[str, Any]] = []
        used_origs = set()

        for i in range(n_o):
            if i in used_origs:
                continue
            s_idxs = orig_to_simps.get(i, [])
            if s_idxs:
                aligned_groups.append({"orig_idxs": [i], "simp_idxs": sorted(s_idxs)})
                used_origs.add(i)
            else:
                # Check if O_i was merged into a simplified clause assigned to a neighbor
                best_j = int(np.argmax(sim_matrix[i]))
                best_sim = float(sim_matrix[i, best_j])
                if best_sim >= 0.55 and aligned_groups and best_j in aligned_groups[-1]["simp_idxs"]:
                    aligned_groups[-1]["orig_idxs"].append(i)
                    used_origs.add(i)
                elif best_sim >= 0.45 and best_j not in simp_to_orig:
                    aligned_groups.append({"orig_idxs": [i], "simp_idxs": [best_j]})
                    simp_to_orig[best_j] = i
                    used_origs.add(i)
                else:
                    # Genuinely omitted clause!
                    aligned_groups.append({"orig_idxs": [i], "simp_idxs": []})
                    used_origs.add(i)

        extra_simps = [j for j in range(n_s) if j not in simp_to_orig]
        return aligned_groups, extra_simps

    @classmethod
    def evaluate_document(cls, engine, orig: str, simp: str, domain: str = "auto") -> Dict[str, Any]:
        """
        Evaluates an entire multi-clause / multi-paragraph contract or document.
        Resolves definitions & pronouns sequentially, aligns clauses cleanly, and
        produces a holistic document scorecard alongside clause-by-clause diagnostics.
        """
        orig_clauses = cls.split_clauses(orig)
        simp_clauses = cls.split_clauses(simp)

        # Edge fallback: if single clause each, delegate seamlessly
        if len(orig_clauses) <= 1 and len(simp_clauses) <= 1:
            single_res = engine.evaluate_pair(orig, simp, domain=domain)
            single_res["is_document"] = False
            return single_res

        # 1. Sequential Pronoun & Defined-Term Resolution
        mem_orig = DocumentContextMemory()
        orig_resolved: List[str] = []
        orig_notes: List[List[str]] = []
        for c in orig_clauses:
            res_c, notes = mem_orig.resolve_clause(c)
            orig_resolved.append(res_c)
            orig_notes.append(notes)

        # Seed simplification memory with defined aliases discovered in original doc
        # so "Vendor" simplifies consistently even if the target omitted `(hereinafter "Vendor")`
        mem_simp = DocumentContextMemory(shared_aliases=mem_orig.aliases)
        simp_resolved: List[str] = []
        simp_notes: List[List[str]] = []
        for c in simp_clauses:
            res_c, notes = mem_simp.resolve_clause(c)
            simp_resolved.append(res_c)
            simp_notes.append(notes)

        # 2. Align Clauses Semantically
        groups, extra_simp_idxs = cls._align_clauses(orig_resolved, simp_resolved, engine)

        # 3. Evaluate Every Aligned Group
        clause_results: List[Dict[str, Any]] = []
        all_resolutions: List[str] = []
        weighted_sum = 0.0
        total_weight = 0.0
        omitted_count = 0
        contradiction_count = 0
        warning_count = 0
        preserved_count = 0

        for idx, grp in enumerate(groups, 1):
            o_idxs = grp["orig_idxs"]
            s_idxs = grp["simp_idxs"]

            raw_orig_grp = " ".join(orig_clauses[i] for i in o_idxs)
            res_orig_grp = " ".join(orig_resolved[i] for i in o_idxs)
            grp_notes: List[str] = []
            for i in o_idxs:
                grp_notes.extend(orig_notes[i])
            for j in s_idxs:
                grp_notes.extend(simp_notes[j])
            # Deduplicate notes preserving order
            seen_n = set()
            unique_notes = []
            for n in grp_notes:
                if n not in seen_n:
                    seen_n.add(n)
                    unique_notes.append(n)
                    if n not in all_resolutions:
                        all_resolutions.append(n)

            clause_weight = max(1.0, float(len(raw_orig_grp.split())))

            if not s_idxs:
                omitted_count += 1
                c_score = 0.0
                c_status = "omitted"
                c_tier = "Omitted Source Clause"
                c_alerts = [{
                    "type": "omitted_clause",
                    "level": "danger",
                    "message": "Entire source clause was omitted in the simplified document.",
                }]
                raw_simp_grp = "(Omitted / Missing in target)"
                res_simp_grp = ""
                nli_verdict = "Missing Clause (Unilateral Information Loss)"
            else:
                raw_simp_grp = " ".join(simp_clauses[j] for j in s_idxs)
                res_simp_grp = " ".join(simp_resolved[j] for j in s_idxs)
                ev = engine.evaluate_pair(res_orig_grp, res_simp_grp, domain=domain)
                c_score = float(ev.get("score", 0.0))
                c_tier = ev.get("tier", "")
                c_alerts = ev.get("guardrails", {}).get("alerts", [])
                nli_verdict = ev.get("nli", {}).get("verdict", "")
                passed = ev.get("guardrails", {}).get("passed", True)

                if c_score <= 2.5 or any(a.get("level") == "danger" for a in c_alerts):
                    c_status = "contradiction"
                    contradiction_count += 1
                elif c_score < 7.5 or not passed:
                    c_status = "warning"
                    warning_count += 1
                else:
                    c_status = "preserved"
                    preserved_count += 1

            weighted_sum += c_score * clause_weight
            total_weight += clause_weight

            clause_results.append({
                "clause_num": idx,
                "orig": raw_orig_grp,
                "simp": raw_simp_grp,
                "resolved_orig": res_orig_grp,
                "resolved_simp": res_simp_grp,
                "status": c_status,
                "score": round(c_score, 1),
                "tier": c_tier,
                "nli_verdict": nli_verdict,
                "alerts": c_alerts,
                "resolutions": unique_notes,
            })

        # Penalty for unsubstantiated / fabricated extra clauses in simplification
        fabricated_clauses = [simp_clauses[j] for j in extra_simp_idxs]
        if fabricated_clauses:
            for extra_txt in fabricated_clauses:
                warning_count += 1
                total_weight += 5.0
                clause_results.append({
                    "clause_num": len(clause_results) + 1,
                    "orig": "(Not present in original document)",
                    "simp": extra_txt,
                    "resolved_orig": "",
                    "resolved_simp": extra_txt,
                    "status": "warning",
                    "score": 2.0,
                    "tier": "Unsubstantiated Extra Clause",
                    "nli_verdict": "Hallucinated / Added Clause",
                    "alerts": [{
                        "type": "added_clause",
                        "level": "warning",
                        "message": "Additional clause in simplification does not correspond to any source clause.",
                    }],
                    "resolutions": [],
                })

        # Document-wide score calculation (penalize heavily if critical omissions/contradictions occurred)
        raw_avg = weighted_sum / max(total_weight, 1.0)
        doc_score = raw_avg
        if contradiction_count > 0 or omitted_count > 0:
            penalty_factor = 1.0 - min(0.45, 0.12 * (contradiction_count + omitted_count))
            doc_score = min(doc_score, 7.9) * penalty_factor
        doc_score = round(max(0.0, min(10.0, doc_score)), 1)

        if doc_score >= 8.5 and contradiction_count == 0 and omitted_count == 0:
            tier = "Excellent Document Fidelity"
            tier_class = "success"
            interp = (
                f"Whole-document audit verified {preserved_count} of {len(groups)} clauses with "
                f"complete semantic equivalence and zero omissions."
            )
        elif doc_score >= 6.5 and contradiction_count == 0 and omitted_count == 0:
            tier = "Good Document Fidelity (Minor Nuances)"
            tier_class = "info"
            interp = (
                f"Audited {len(groups)} clauses across document. Core meaning preserved with "
                f"{warning_count} minor nuance alert(s)."
            )
        elif contradiction_count > 0 or omitted_count > 0:
            tier = "Severe Document Risk (Contradictions / Omissions)"
            tier_class = "danger"
            interp = (
                f"⚠️ Whole-document audit detected {contradiction_count} contradiction(s) and "
                f"{omitted_count} omitted source clause(s) across {len(groups)} clauses."
            )
        else:
            tier = "Moderate Document Drift"
            tier_class = "warning"
            interp = f"Audited {len(groups)} clauses; {warning_count} clause(s) require manual review."

        # Detect aggregate language pair across document & build unified response
        lang_o = engine.detect_lang(orig)
        lang_s = engine.detect_lang(simp)

        doc_alerts = [a for cr in clause_results for a in cr.get("alerts", [])]
        doc_passed = (contradiction_count == 0 and omitted_count == 0 and len(doc_alerts) == 0)
        words_o = len(orig.split())
        words_s = len(simp.split())
        comp_ratio = round((1.0 - (len(simp) / max(len(orig), 1))) * 100, 1)

        from demo_app.legal_guardrails import LegalGuardrails
        resolved_domain = domain if domain != "auto" else LegalGuardrails.detect_domain(orig + " " + simp)

        return {
            "is_document": True,
            "score": doc_score,
            "tier": tier,
            "tier_class": tier_class,
            "interpretation": interp,
            "mode": f"Whole-Document Auditor + Coreference Memory ({lang_o.upper()}↔{lang_s.upper()})",
            "domain": resolved_domain,
            "predicted_category": f"Whole Document ({len(groups)} Clauses)",
            "lang_orig": lang_o,
            "lang_simp": lang_s,
            "guardrails": {
                "passed": doc_passed,
                "alerts": doc_alerts[:6],
            },
            "nli": {
                "verdict": f"Document Audit ({preserved_count}/{len(groups)} Preserved)",
            },
            "metrics": {
                "colbert_maxsim": round(doc_score * 10, 1),
                "cosine_similarity": round(doc_score * 10, 1),
                "jaccard": 0.0,
                "compression_ratio": comp_ratio,
                "words_orig": words_o,
                "words_simp": words_s,
            },
            "summary": {
                "total_orig_clauses": len(orig_clauses),
                "total_simp_clauses": len(simp_clauses),
                "aligned_groups": len(groups),
                "preserved": preserved_count,
                "warnings": warning_count,
                "contradictions": contradiction_count,
                "omitted": omitted_count,
                "added_extra": len(fabricated_clauses),
            },
            "resolved_references": all_resolutions,
            "clauses": clause_results,
        }

    # Explicit page number / watermark / divider line patterns stripped during document import
    _NOISE_LINE_PATTERNS = [
        re.compile(r"^\s*page\s+\d+(?:\s*(?:of|/)\s*\d+)?(?:\s*[-|•].*)?$", re.IGNORECASE),
        re.compile(r"^\s*(?:p\.?|pg\.?)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*[-–—|•\s]*\d{1,4}\s*(?:of|/)\s*\d{1,4}[-–—|•\s]*$", re.IGNORECASE),
        re.compile(r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$"),
        re.compile(r"^\s*\d{1,4}\s*$"),
        re.compile(r"^\s*(?:strictly\s+)?(?:private\s+and\s+|privileged\s*(?:&|and)\s*)?confidential(?:\s*draft|\s*copy)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*(?:draft|execution\s+(?:copy|version)|for\s+(?:internal\s+use|discussion\s+purposes)\s+only|do\s+not\s+distribute)\s*$", re.IGNORECASE),
        re.compile(r"^\s*(?:©|copyright)\s*\d{4}\b.*all\s+rights\s+reserved\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*[_\-=~*]{4,}\s*$"),
    ]

    _TRUE_HYPHEN_PREFIXES = {
        "non", "pre", "post", "co", "re", "sub", "anti", "self", "multi",
        "cross", "ex", "vice", "semi", "inter", "counter", "over", "under",
    }

    _STRUCTURAL_START_RE = re.compile(
        r"^(?:\d{1,3}[\.\)]\s+|\([a-zA-Z0-9]{1,3}\)\s+|[a-zA-Z]\)\s+|(?:Section|Clause|Article|Schedule|Annex|Exhibit)\s+\d+|[•▪◦\-\*]\s+)",
        re.IGNORECASE,
    )

    @classmethod
    def _is_noise_header_footer_line(cls, line: str) -> bool:
        """Detects page numbers, watermarks, copyright footers, and decorative dividers."""
        s = re.sub(r"[ \t]+", " ", line or "").strip()
        if not s:
            return True
        if any(p.match(s) for p in cls._NOISE_LINE_PATTERNS):
            return True
        # Catch combined footer lines like "Acme Corp Contract | Page 2 of 10" or "CONFIDENTIAL - Page 3"
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", s, re.IGNORECASE) and len(s) <= 100:
            return True
        if re.search(r"(?:confidential|draft)\s*[-|•]\s*page\s+\d+", s, re.IGNORECASE) and len(s) <= 100:
            return True
        return False

    @classmethod
    def _detect_repeating_edge_headers(cls, page_lines_list: List[List[str]]) -> set:
        """
        Detects custom running headers/footers repeated across top/bottom 2 lines
        of >= 2 pages in multi-page PDFs (normalizing changing page numbers to '#').
        """
        if len(page_lines_list) < 2:
            return set()

        from collections import Counter
        sig_counts: Counter = Counter()
        for p_lines in page_lines_list:
            clean_p = [re.sub(r"[ \t]+", " ", ln).strip() for ln in p_lines if ln.strip()]
            if not clean_p:
                continue
            edge_candidates = clean_p[:2] + (clean_p[-2:] if len(clean_p) > 2 else [])
            seen_on_page = set()
            for cand in edge_candidates:
                if len(cand) > 95:
                    continue
                sig = re.sub(r"\d+", "#", cand.lower()).strip()
                if sig and sig not in seen_on_page:
                    sig_counts[sig] += 1
                    seen_on_page.add(sig)

        min_pages = max(2, int(len(page_lines_list) * 0.4))
        return {sig for sig, cnt in sig_counts.items() if cnt >= min_pages}

    @classmethod
    def _deinterleave_layout_lines(cls, layout_lines: List[str]) -> List[str]:
        """
        Detects 2-column prose blocks in PDF layout text (where lines consistently
        split at a shared wide horizontal whitespace gutter) and reorders them as
        [Left Column Top-to-Bottom] followed by [Right Column Top-to-Bottom].
        Also formats wide single-row table gaps cleanly as `Key: Value.`.
        """
        if not layout_lines:
            return []

        def find_best_gutter(ln: str) -> Optional[Tuple[int, str, str]]:
            # Look for >= 4 consecutive spaces roughly in the central 20%-80% band
            best = None
            for m in re.finditer(r"\S(?:\s{4,})\S", ln):
                gap_start = m.start() + 1
                gap_end = m.end() - 1
                left = ln[:gap_start].strip()
                right = ln[gap_end:].strip()
                if left and right:
                    mid = (gap_start + gap_end) // 2
                    if 0.15 * len(ln) <= mid <= 0.85 * len(ln):
                        best = (mid, left, right)
                        break
            return best

        parsed_rows = [(ln, find_best_gutter(ln)) for ln in layout_lines]
        output: List[str] = []
        i = 0
        n = len(parsed_rows)

        while i < n:
            ln, g = parsed_rows[i]
            if g is None:
                cleaned = re.sub(r"[ \t]+", " ", ln).strip()
                if cleaned:
                    output.append(cleaned)
                i += 1
                continue

            # Collect consecutive rows with a roughly aligned gutter (within +/- 8 chars)
            base_mid = g[0]
            block = [g]
            j = i + 1
            while j < n and parsed_rows[j][1] is not None and abs(parsed_rows[j][1][0] - base_mid) <= 8:
                block.append(parsed_rows[j][1])
                j += 1

            lefts = [b[1] for b in block]
            rights = [b[2] for b in block]
            avg_left_words = sum(len(x.split()) for x in lefts) / len(lefts)
            avg_right_words = sum(len(x.split()) for x in rights) / len(rights)

            # Multi-line narrative block (>= 3 rows OR multi-word prose on both sides): treat as 2 columns!
            if len(block) >= 3 and avg_left_words >= 3.0 and avg_right_words >= 3.0:
                output.extend(lefts)
                output.extend(rights)
            else:
                # Key-value table rows (e.g. "Governing Law    State of Delaware")
                for left, right in zip(lefts, rights):
                    l_clean = left.rstrip(":").strip()
                    if len(l_clean.split()) <= 5 and not l_clean.endswith((".", "!", "?")):
                        row_txt = f"{l_clean}: {right}"
                        if not row_txt.endswith((".", "!", "?", "।")):
                            row_txt += "."
                        output.append(row_txt)
                    else:
                        output.append(f"{left} {right}")
            i = j

        return output

    @classmethod
    def _clean_and_unwrap_lines(cls, lines: List[str]) -> str:
        """
        Reconstructs true paragraphs/clauses from wrapped PDF lines:
        - Fixes end-of-line hyphenation (e.g. 'agree-\\nment' -> 'agreement').
        - Merges wrapped continuation lines when previous line lacks terminal punctuation.
        - Preserves true section headings, list items, and sentence endings.
        """
        cleaned_lines = [re.sub(r"[ \t]+", " ", re.sub(r"[\uf0b7•▪◦]", "• ", ln)).strip() for ln in lines if ln and ln.strip()]
        if not cleaned_lines:
            return ""

        paragraphs: List[str] = []
        buf = ""

        for ln in cleaned_lines:
            if not buf:
                buf = ln
                continue

            ends_terminal = bool(re.search(r"[.!?:;।\"”\)]$", buf))
            starts_structural = bool(cls._STRUCTURAL_START_RE.match(ln))
            is_short_heading = len(buf.split()) <= 5 and buf.isupper()

            # Check hyphenation split across line boundary
            hyph_m = re.search(r"([A-Za-z]{2,})-$", buf)
            next_word_m = re.match(r"^([a-z]{2,})", ln)
            if hyph_m and next_word_m and not starts_structural:
                prefix = hyph_m.group(1).lower()
                if prefix in cls._TRUE_HYPHEN_PREFIXES:
                    buf = buf + ln  # Keep genuine compound hyphen (e.g. pre-approved)
                else:
                    buf = buf[:-1] + ln  # Join broken word (e.g. agree- + ment -> agreement)
                continue

            if ends_terminal or starts_structural or is_short_heading:
                paragraphs.append(buf.strip())
                buf = ln
            else:
                buf = f"{buf} {ln}"

        if buf:
            paragraphs.append(buf.strip())

        return "\n".join(paragraphs)

    @classmethod
    def _extract_pdf_text(cls, raw_bytes: bytes) -> str:
        """Extracts noise-free, de-columnized, unwrapped text from PDF bytes."""
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))

        page_lines_list: List[List[str]] = []
        for page in reader.pages:
            raw_txt = ""
            try:
                raw_txt = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                raw_txt = ""
            if not raw_txt.strip():
                raw_txt = page.extract_text() or ""
            lines = [ln.rstrip() for ln in raw_txt.replace("\r\n", "\n").splitlines() if ln.strip()]
            if lines:
                page_lines_list.append(lines)

        if not page_lines_list:
            return ""

        repeating_sigs = cls._detect_repeating_edge_headers(page_lines_list)

        all_clean_lines: List[str] = []
        for p_lines in page_lines_list:
            filtered_layout: List[str] = []
            for ln in p_lines:
                collapsed = re.sub(r"[ \t]+", " ", ln).strip()
                if cls._is_noise_header_footer_line(collapsed):
                    continue
                sig = re.sub(r"\d+", "#", collapsed.lower()).strip()
                if sig in repeating_sigs:
                    continue
                filtered_layout.append(ln)

            deinterleaved = cls._deinterleave_layout_lines(filtered_layout)
            all_clean_lines.extend(deinterleaved)

        return cls._clean_and_unwrap_lines(all_clean_lines)

    @classmethod
    def _extract_docx_text(cls, raw_bytes: bytes) -> str:
        """Extracts noise-free text and deduplicated table rows from DOCX bytes."""
        import docx
        doc = docx.Document(io.BytesIO(raw_bytes))
        blocks: List[str] = []

        for p in doc.paragraphs:
            txt = re.sub(r"[ \t]+", " ", p.text).strip()
            if txt and not cls._is_noise_header_footer_line(txt):
                blocks.append(txt)

        for tbl in doc.tables:
            for row in tbl.rows:
                # Deduplicate merged XML table cells cleanly
                seen_tcs = set()
                cells: List[str] = []
                for c in row.cells:
                    tc_id = id(c._tc)
                    if tc_id in seen_tcs:
                        continue
                    seen_tcs.add(tc_id)
                    c_txt = re.sub(r"\s+", " ", c.text).strip()
                    if c_txt and not cls._is_noise_header_footer_line(c_txt):
                        cells.append(c_txt)
                if not cells:
                    continue
                if len(cells) == 2 and len(cells[0].split()) <= 5 and not cells[0].endswith((".", "!", "?")):
                    row_str = f"{cells[0].rstrip(':')}: {cells[1]}"
                    if not row_str.endswith((".", "!", "?", "।")):
                        row_str += "."
                    blocks.append(row_str)
                else:
                    blocks.append(" ".join(cells))

        return "\n".join(blocks)

    @classmethod
    def extract_text_from_file(cls, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts clean plain text from uploaded .txt, .md, .pdf, or .docx files
        for whole-document evaluation. Automatically strips PDF running headers/footers,
        de-interleaves multi-column layouts, unwraps mid-sentence breaks, and cleans tables.
        """
        if not filename:
            raise ValueError("Missing filename.")
        if not raw_bytes:
            raise ValueError("Uploaded file is empty.")
        if len(raw_bytes) > 5 * 1024 * 1024:
            raise ValueError("File exceeds maximum allowed size of 5 MB.")

        ext = os.path.splitext(filename.lower())[1]
        allowed = {".txt", ".md", ".pdf", ".docx"}
        if ext not in allowed:
            raise ValueError(f"Unsupported file type '{ext}'. Supported types: .txt, .md, .pdf, .docx")

        clean_text = ""
        if ext in (".txt", ".md"):
            try:
                raw_txt = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_txt = raw_bytes.decode("cp1252", errors="replace")
            raw_lines = [ln for ln in raw_txt.replace("\r\n", "\n").splitlines() if not cls._is_noise_header_footer_line(ln)]
            clean_text = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in raw_lines if ln.strip())
        elif ext == ".pdf":
            try:
                clean_text = cls._extract_pdf_text(raw_bytes)
            except Exception as exc:
                raise ValueError(f"Could not read PDF file: {exc}")
            if not clean_text.strip():
                raise ValueError("No selectable text found in PDF (image-only scanned PDFs require OCR).")
        elif ext == ".docx":
            try:
                clean_text = cls._extract_docx_text(raw_bytes)
            except Exception as exc:
                raise ValueError(f"Could not read DOCX file: {exc}")

        if not clean_text.strip():
            raise ValueError("Document contains no readable text.")

        clauses = cls.split_clauses(clean_text)
        return {
            "success": True,
            "filename": os.path.basename(filename),
            "ext": ext,
            "text": clean_text,
            "char_count": len(clean_text),
            "clause_count": len(clauses),
        }
