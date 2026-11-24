import json
import urllib.request
import urllib.error
import re
from typing import List, Dict, Any, Optional

class GeminiGenerator:
    """
    Gemini Generative Plain-Language Author for JUDGEBERT.
    
    Generates fluent, natural plain-language legal reformulations from scratch,
    which are then evaluated and verified through JUDGEBERT's Closed-Loop Arbiter.
    """
    
    MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    @classmethod
    def generate_simplification_candidates(
        cls,
        orig: str,
        flawed_simp: str,
        alerts: List[Dict[str, Any]],
        lang: str = "en",
        api_key: Optional[str] = None,
        count: int = 3
    ) -> List[str]:
        """
        Calls the Google Gemini API to generate fluent plain-language reformulations.
        Returns a list of clean candidate strings, or an empty list if API call fails.
        """
        if not api_key or not api_key.strip():
            return []
            
        key = api_key.strip()
        alert_summaries = [a.get("message", "") for a in alerts if a.get("message")]
        alert_context = " | ".join(alert_summaries) if alert_summaries else "No specific guardrail alerts."
        
        lang_name = "English"
        if lang == "fr":
            lang_name = "French"
        elif lang == "hi":
            lang_name = "Hindi (Devanagari script)"

        system_prompt = (
            f"You are an expert legal drafter and plain-language communication specialist fluent in {lang_name}.\n"
            f"Your mission is to rewrite a complex legal sentence into clean, professional, and easily understandable {lang_name} plain language.\n\n"
            f"CRITICAL LEGAL PRESERVATION RULES:\n"
            f"1. You must preserve 100% of the legal rights, obligations, prohibitions, conditions, and exceptions from the Original.\n"
            f"2. Never invert actors or parties (e.g., if Party A must pay Party B, never make Party B pay Party A).\n"
            f"3. Strictly preserve all dates, numbers, timeframes, and monetary amounts without alteration.\n"
            f"4. If the user's attempt had errors ({alert_context}), fix them completely.\n"
            f"5. The rewritten output must be in natural, everyday {lang_name} that a non-lawyer can effortlessly understand.\n"
            f"6. Do NOT copy the original sentence verbatim. Rephrase it with simpler, modern vocabulary.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Respond ONLY with a valid JSON array containing exactly {count} distinct plain-language candidates, for example:\n"
            f'["First distinct plain-language version.", "Second distinct plain-language version.", "Third distinct plain-language version."]'
        )
        
        user_content = (
            f"ORIGINAL LEGAL CLAUSE:\n\"{orig}\"\n\n"
            f"FLAWED SIMPLIFICATION ATTEMPT:\n\"{flawed_simp}\"\n\n"
            f"DETECTED FLAWS TO FIX:\n{alert_context}\n\n"
            f"Provide {count} distinct, faithful {lang_name} plain-language reformulations:"
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
                "temperature": 0.3,
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
                                    print(f"[INFO] Successfully generated {len(parsed)} candidates via Gemini ({model})")
                                    return parsed
            except urllib.error.HTTPError as eh:
                print(f"[WARN] Gemini API error ({model}): HTTP {eh.code} - {eh.reason}")
                if eh.code in (400, 403):
                    # Invalid key or blocked, no point retrying other models
                    break
            except Exception as e:
                print(f"[WARN] Gemini request failed for {model}: {e}")
                continue
                
        return []

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
