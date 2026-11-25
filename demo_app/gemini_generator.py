import json
import urllib.request
import urllib.error
import re
import time
from typing import List, Dict, Any, Optional
from demo_app.privacy_masker import PrivacyMasker, MaskingSession


class GeminiGenerator:
    """
    Gemini Generative Plain-Language Author for JUDGEBERT.

    Generates fluent, natural plain-language legal reformulations from scratch,
    which are then evaluated and verified through JUDGEBERT's Closed-Loop Arbiter.
    Supports English, French, and Hindi (Devanagari script), with built-in local
    PII / entity masking before external cloud API transmission.
    """

    MODELS = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemma-4-26b-a4b-it",
    ]

    _UNRESOLVED_TAG_RE = re.compile(
        r"\[(?:ORG|PERSON|AMOUNT|PERCENT|DATE|EMAIL|PHONE|URL|ID|ADDRESS|PROJECT|NUM)_\d+\]",
        re.IGNORECASE,
    )

    @classmethod
    def generate_simplification_candidates(
        cls,
        orig: str,
        flawed_simp: str,
        alerts: List[Dict[str, Any]],
        lang: str = "en",
        api_key: Optional[str] = None,
        count: int = 3,
        mask_pii: bool = True,
    ) -> List[str]:
        """
        Calls the Google Gemini API (with local PII redaction enabled by default)
        to generate fluent plain-language reformulations in the requested language.
        """
        res = cls.generate_simplification_candidates_with_audit(
            orig=orig,
            flawed_simp=flawed_simp,
            alerts=alerts,
            lang=lang,
            api_key=api_key,
            count=count,
            mask_pii=mask_pii,
        )
        return res.get("candidates", [])

    @classmethod
    def generate_simplification_candidates_with_audit(
        cls,
        orig: str,
        flawed_simp: str,
        alerts: List[Dict[str, Any]],
        lang: str = "en",
        api_key: Optional[str] = None,
        count: int = 3,
        mask_pii: bool = True,
    ) -> Dict[str, Any]:
        """
        Scrubs sensitive PII/entities locally before contacting Gemini, restores real
        entities after generation, and returns candidates alongside a privacy audit dict.
        """
        empty_audit: Dict[str, Any] = {
            "candidates": [],
            "privacy_masked": False,
            "masked_count": 0,
            "masked_summary": "",
            "masked_orig": orig,
        }
        if not api_key or not api_key.strip():
            return empty_audit

        key = api_key.strip()

        # Scrub sensitive data locally prior to prompt assembly
        session: Optional[MaskingSession] = None
        if mask_pii:
            send_orig, send_simp, send_alerts, session = PrivacyMasker.mask_inputs(
                orig, flawed_simp, alerts
            )
            if session.masked_count > 0:
                print(
                    f"[INFO] Privacy Shield masked {session.masked_count} item(s) "
                    f"({session.summary_description()}) locally before cloud request."
                )
        else:
            send_orig, send_simp, send_alerts = orig, flawed_simp, alerts

        alert_summaries = [a.get("message", "") for a in (send_alerts or []) if a.get("message")]
        alert_context = " | ".join(alert_summaries) if alert_summaries else "No specific guardrail alerts."

        if lang == "hi":
            lang_name = "Hindi (Devanagari script हिन्दी)"
            lang_guidance = (
                "Write the simplification entirely in natural, professional, and clear Hindi using Devanagari script (हिन्दी).\n"
                "Use everyday, easily understood vocabulary while strictly maintaining legal accuracy.\n"
                "Example format: [\"किराएदार को हर महीने की पहली तारीख को पूरा किराया देना होगा।\", \"...\"]"
            )
        elif lang == "fr":
            lang_name = "French (Français)"
            lang_guidance = (
                "Write the simplification entirely in natural, fluent, and professional French (Français).\n"
                "Use modern, clear vocabulary that a French speaker can easily understand without legal jargon.\n"
                "Example format: [\"Le locataire doit verser l'intégralité du loyer le premier de chaque mois.\", \"...\"]"
            )
        else:
            lang_name = "Plain English"
            lang_guidance = (
                "Write the simplification entirely in natural, fluent Plain English.\n"
                "Use modern, clear everyday vocabulary that a consumer can effortlessly understand.\n"
                "Example format: [\"The tenant must pay rent in full on the first of each month.\", \"...\"]"
            )

        placeholder_rule = ""
        if session and session.masked_count > 0:
            tokens_preview = ", ".join(list(session.token_to_value.keys())[:10])
            placeholder_rule = (
                f"\n7. PRIVACY PLACEHOLDERS PRESENT ({tokens_preview}): Preserve every bracketed placeholder token "
                f"from the Original verbatim (exact uppercase tag and square brackets) in all candidates. Never translate, rename, or omit them."
            )

        system_prompt = (
            f"You are an expert legal drafter, translator, and plain-language communication specialist fluent in {lang_name}.\n"
            f"Your mission is to rewrite a legal sentence into clean, professional, and easily understandable {lang_name} plain language.\n\n"
            f"LANGUAGE SPECIFICATION:\n"
            f"{lang_guidance}\n\n"
            f"CRITICAL LEGAL PRESERVATION RULES:\n"
            f"1. You must preserve 100% of the legal rights, obligations, prohibitions, conditions, and exceptions from the Original.\n"
            f"2. Never invert actors or parties (e.g., if Party A must pay Party B, never make Party B pay Party A).\n"
            f"3. Strictly preserve all dates, numbers, timeframes, and monetary amounts without alteration.\n"
            f"4. If the user's attempt had errors ({alert_context}), fix them completely.\n"
            f"5. The rewritten output must be in natural {lang_name}.\n"
            f"6. Do NOT copy the original sentence verbatim. Rephrase it with simpler, modern vocabulary in {lang_name}.{placeholder_rule}\n\n"
            f"OUTPUT FORMAT:\n"
            f"Respond ONLY with a valid JSON array containing exactly {count} distinct plain-language candidates in {lang_name}, for example:\n"
            f'["First distinct plain-language version.", "Second distinct plain-language version.", "Third distinct plain-language version."]'
        )

        user_content = (
            f"ORIGINAL LEGAL CLAUSE:\n\"{send_orig}\"\n\n"
            f"CURRENT ATTEMPT:\n\"{send_simp}\"\n\n"
            f"TARGET LANGUAGE REQUIRED: {lang_name}\n"
            f"DETECTED FLAWS TO FIX:\n{alert_context}\n\n"
            f"Provide {count} distinct, faithful {lang_name} plain-language reformulations as a JSON array of strings:"
        )

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt + "\n\n" + user_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.25,
                "topP": 0.85,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json"
            }
        }

        payload_bytes = json.dumps(body).encode("utf-8")

        for model in cls.MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            candidates_raw = data.get("candidates", [])
                            if candidates_raw:
                                content = candidates_raw[0].get("content", {})
                                parts = content.get("parts", [])
                                if parts:
                                    raw_text = parts[0].get("text", "").strip()
                                    parsed = cls._parse_json_candidates(raw_text)
                                    if parsed:
                                        restored: List[str] = []
                                        for cand in parsed:
                                            unmasked = PrivacyMasker.unmask_text(cand, session) if session else cand
                                            # Drop malformed candidates with leftover invented placeholder tags
                                            if cls._UNRESOLVED_TAG_RE.search(unmasked):
                                                continue
                                            restored.append(unmasked)
                                        if restored:
                                            print(f"[INFO] Successfully generated {len(restored)} candidates in {lang_name} via Gemini ({model})")
                                            return {
                                                "candidates": restored,
                                                "privacy_masked": bool(session and session.masked_count > 0),
                                                "masked_count": session.masked_count if session else 0,
                                                "masked_summary": session.summary_description() if session else "",
                                                "masked_orig": send_orig,
                                                "model": model,
                                            }
                    break
                except urllib.error.HTTPError as eh:
                    print(f"[WARN] Gemini API error ({model}): HTTP {eh.code} - {eh.reason}")
                    if eh.code in (400, 403):
                        return empty_audit
                    if eh.code in (429, 503) and attempt == 0:
                        time.sleep(1.2)
                        continue
                    break
                except Exception as e:
                    print(f"[WARN] Gemini request failed for {model}: {e}")
                    break

        return empty_audit

    @classmethod
    def _parse_json_candidates(cls, text: str) -> List[str]:
        """Safely parses JSON array of candidates from model response."""
        try:
            val = json.loads(text)
            if isinstance(val, list):
                return [str(item).strip() for item in val if str(item).strip()]
            if isinstance(val, dict):
                for k in ["candidates", "simplifications", "results", "options"]:
                    if k in val and isinstance(val[k], list):
                        return [str(item).strip() for item in val[k] if str(item).strip()]
        except Exception:
            pass
            
        # Fallback regex extraction if model wrapped in markdown code fence
        match = re.search(r"\[\s*\"(?:\\.|[^\"])*\"(?:\s*,\s*\"(?:\\.|[^\"])*\")*\s*\]", text, re.DOTALL)
        if match:
            try:
                val = json.loads(match.group(0))
                if isinstance(val, list):
                    return [str(item).strip() for item in val if str(item).strip()]
            except Exception:
                pass
                
        # Line-by-line fallback
        lines = [re.sub(r"^[\d\.\-\*\s\"\']+", "", line).rstrip("\"'").strip() for line in text.splitlines()]
        return [l for l in lines if len(l) > 10 and not l.startswith("[") and not l.startswith("{")]
