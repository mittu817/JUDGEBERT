import re
import json
import random
from pathlib import Path

def clean_lines(text):
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
    merged = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith("Title:") or line.startswith("==="):
            i += 1
            continue
        # If line ends with reporting clause like "said the court:", "अदालत ने कहा,"
        if i + 1 < len(raw_lines) and (
            line.endswith(":") or line.endswith(",") or 
            line.endswith("अदालत ने कहा,") or line.endswith("कोर्ट ने कहा,") or
            line.endswith("सेंघ ने प्रस्तुत किया,") or line.endswith("वकील ने कहा,") or
            line.endswith("the court said:") or line.endswith("submitted:") or line.endswith("observed:")
        ):
            merged.append(line + " " + raw_lines[i+1])
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged

def invert_hindi_legal(text):
    """Generates hard legal contradictions by inverting legal outcomes and deontic forces."""
    substitutions = [
        ("जमानत देने से इनकार करते हुए", "जमानत स्वीकार करते हुए"),
        ("जमानत खारिज करते हुए", "जमानत स्वीकार करते हुए"),
        ("जमानत देते हुए", "जमानत देने से इनकार करते हुए"),
        ("जमानत दे दी", "जमानत याचिका खारिज कर दी"),
        ("अग्रिम जमानत देने से इनकार", "अग्रिम जमानत प्रदान"),
        ("लागू नहीं होगी", "अनिवार्य रूप से लागू होगी"),
        ("लागू होगी", "किसी भी स्थिति में लागू नहीं होगी"),
        ("भुगतान करने का निर्देश दिया", "भुगतान करने के दावे को खारिज कर दिया"),
        ("तलाक ‌की डिक्री प्रदान की", "तलाक की याचिका खारिज कर दी"),
        ("डिक्री प्रदान की", "दावे को खारिज कर दिया"),
        ("अनुमति नहीं दी जा सकती", "पूर्ण अनुमति प्रदान की जाती है"),
        ("क्रूरता नहीं है", "गंभीर क्रूरता माना जाता है"),
        ("क्रूरता नहीं माना जा सकता", "कानूनन गंभीर क्रूरता माना जाएगा"),
        ("अवज्ञा नहीं माना जा सकता", "न्यायालय की घोर अवमानना माना जाएगा"),
        ("दोषी नहीं माना", "पूरी तरह से दोषी करार दिया"),
        ("दोषमुक्त किया", "दोषी ठहराया गया"),
        ("अधिकार होगा", "कोई कानूनी अधिकार नहीं होगा"),
    ]
    for orig, rep in substitutions:
        if orig in text:
            return text.replace(orig, rep, 1), f"Flipped outcome: '{orig}' -> '{rep}'"
    
    # Generic negation inversion
    if " नहीं " in text:
        return text.replace(" नहीं ", " अवश्य ", 1), "Removed negation"
    elif " किया गया" in text:
        return text.replace(" किया गया", " नहीं किया गया", 1), "Added negation"
    elif " दिया गया" in text:
        return text.replace(" दिया गया", " नहीं दिया गया", 1), "Added negation"
    return None, None

def process_all_samples():
    base_dir = Path("datastore/cross_lingual/mildsum")
    samples_dir = base_dir / "samples"
    
    en_hi_pairs = []
    hi_hi_pairs = []
    contra_pairs = []
    all_benchmark_pairs = []
    
    pair_id = 1
    
    for s_idx in range(1, 11):
        s_dir = samples_dir / f"Sample_{s_idx}"
        if not s_dir.exists():
            continue
        
        en_text = (s_dir / "EN_Summary.txt").read_text(encoding="utf-8")
        hi_text = (s_dir / "HI_Summary.txt").read_text(encoding="utf-8")
        
        en_lines = clean_lines(en_text)
        hi_lines = clean_lines(hi_text)
        
        # Match sentences where length is close
        min_len = min(len(en_lines), len(hi_lines))
        for j in range(min_len):
            en_sent = en_lines[j].strip().strip('"').strip("'")
            hi_sent = hi_lines[j].strip().strip('"').strip("'")
            
            if len(en_sent) < 25 or len(hi_sent) < 20:
                continue
            
            # Cross-lingual Entailment pair
            en_hi_item = {
                "pairID": f"mildsum_en_hi_{pair_id}",
                "sample": f"Sample_{s_idx}",
                "original": en_sent,
                "simplification": hi_sent,
                "score": 10.0,
                "label": "entailment",
                "src_lang": "en",
                "tgt_lang": "hi",
                "domain": "legal_judgment",
                "source": "MILDSum"
            }
            en_hi_pairs.append(en_hi_item)
            all_benchmark_pairs.append(en_hi_item)
            
            # Invert to create hard contradiction
            contra_hi, reason = invert_hindi_legal(hi_sent)
            if contra_hi:
                c_item = {
                    "pairID": f"mildsum_contra_{pair_id}",
                    "sample": f"Sample_{s_idx}",
                    "original": en_sent,
                    "simplification": contra_hi,
                    "score": 0.0,
                    "label": "contradiction",
                    "src_lang": "en",
                    "tgt_lang": "hi",
                    "domain": "legal_judgment",
                    "source": "MILDSum",
                    "perturbation": reason
                }
                contra_pairs.append(c_item)
                all_benchmark_pairs.append(c_item)
                
                # Hindi-Hindi contradiction
                c_hi_item = {
                    "pairID": f"mildsum_hi_contra_{pair_id}",
                    "sample": f"Sample_{s_idx}",
                    "original": hi_sent,
                    "simplification": contra_hi,
                    "score": 0.0,
                    "label": "contradiction",
                    "src_lang": "hi",
                    "tgt_lang": "hi",
                    "domain": "legal_judgment",
                    "source": "MILDSum",
                    "perturbation": reason
                }
                contra_pairs.append(c_hi_item)
                all_benchmark_pairs.append(c_hi_item)
            
            # Monolingual Hindi paraphrase / simplification pair
            # We pair the full legal sentence with a condensed core clause
            words = hi_sent.split()
            if len(words) > 12:
                # Retain core meaning clause
                condensed = " ".join(words[:18])
                if not condensed.endswith("।") and not condensed.endswith("."):
                    condensed += "।"
                hi_hi_item = {
                    "pairID": f"mildsum_hi_hi_{pair_id}",
                    "sample": f"Sample_{s_idx}",
                    "original": hi_sent,
                    "simplification": condensed,
                    "score": 8.5,
                    "label": "entailment",
                    "src_lang": "hi",
                    "tgt_lang": "hi",
                    "domain": "legal_judgment",
                    "source": "MILDSum"
                }
                hi_hi_pairs.append(hi_hi_item)
                all_benchmark_pairs.append(hi_hi_item)
            
            pair_id += 1

    # Split into Train (80%) and Test (20%)
    random.seed(42)
    shuffled = list(all_benchmark_pairs)
    random.shuffle(shuffled)
    split_idx = int(0.8 * len(shuffled))
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]

    # Write files
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items)} items to {path}")

    write_jsonl(base_dir / "mildsum_en_hi_pairs.jsonl", en_hi_pairs)
    write_jsonl(base_dir / "mildsum_hi_hi_pairs.jsonl", hi_hi_pairs)
    write_jsonl(base_dir / "mildsum_contradictions.jsonl", contra_pairs)
    write_jsonl(base_dir / "mildsum_benchmark_all.jsonl", all_benchmark_pairs)
    write_jsonl(base_dir / "train_mildsum.jsonl", train_set)
    write_jsonl(base_dir / "test_mildsum.jsonl", test_set)
    
    print("\n--- Summary of Extracted MILDSum Pairs ---")
    print(f"Cross-lingual EN -> HI pairs: {len(en_hi_pairs)}")
    print(f"Monolingual HI -> HI pairs:   {len(hi_hi_pairs)}")
    print(f"Hard Legal Contradictions:    {len(contra_pairs)}")
    print(f"Total MILDSum Pairs:          {len(all_benchmark_pairs)}")
    print(f"Training split (80%):         {len(train_set)}")
    print(f"Held-out test split (20%):    {len(test_set)}")

if __name__ == "__main__":
    process_all_samples()
