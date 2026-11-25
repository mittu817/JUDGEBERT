import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response

# Add parent directory to path so relative imports work
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from demo_app.model_engine import JudgeBertEngine, CATEGORIES
from demo_app.config import get_gemini_api_key, is_gemini_enabled, is_privacy_masking_enabled

app = Flask(__name__, template_folder=str(current_dir / "templates"))

# Initialize engine singleton
engine = JudgeBertEngine()


@app.route("/")
def index():
    examples = engine.get_benchmark_examples()
    return render_template("index.html", examples=examples, categories=CATEGORIES)


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True) or {}
    orig = data.get("original", "").strip()
    simp = data.get("simplification", "").strip()
    domain = data.get("domain", "auto").strip()
    lang_orig = data.get("lang_orig", "auto").strip().lower()
    lang_simp = data.get("lang_simp", "auto").strip().lower()

    if not orig or not simp:
        return jsonify({"error": "Please provide both the original clause and the simplified version."}), 400

    # Auto-detect Whole-Document multi-clause contracts seamlessly
    from demo_app.document_evaluator import DocumentEvaluator
    if DocumentEvaluator.is_multi_clause_document(orig, simp):
        result = engine.evaluate_document(orig, simp, domain=domain)
        return jsonify(result)

    result = engine.evaluate_pair(orig, simp, domain=domain, lang_orig=lang_orig, lang_simp=lang_simp)
    return jsonify(result)


@app.route("/api/evaluate-document", methods=["POST"])
def evaluate_document():
    data = request.get_json(force=True) or {}
    orig = data.get("original", "").strip()
    simp = data.get("simplification", "").strip()
    domain = data.get("domain", "auto").strip()

    if not orig or not simp:
        return jsonify({"error": "Please provide both the original contract and the simplified version."}), 400

    result = engine.evaluate_document(orig, simp, domain=domain)
    return jsonify(result)


@app.route("/api/extract-document", methods=["POST"])
def extract_document():
    if "file" not in request.files:
        return jsonify({"error": "No document file uploaded."}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty filename received."}), 400

    try:
        from demo_app.document_evaluator import DocumentEvaluator
        raw_bytes = f.read()
        result = DocumentEvaluator.extract_text_from_file(f.filename, raw_bytes)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return jsonify({"error": f"Document processing failed: {exc}"}), 500


@app.route("/api/export-document", methods=["POST"])
def export_document():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    fmt = (data.get("format") or "docx").strip().lower()
    base_name = (data.get("filename") or "judgebert_repaired_document").strip()
    title = (data.get("title") or "Verified Plain-Language Document").strip()
    score = data.get("score")
    tier = (data.get("tier") or "").strip()
    mode = (data.get("mode") or "").strip()

    if not text:
        return jsonify({"error": "No text provided to export."}), 400
    if fmt not in ("docx", "pdf"):
        return jsonify({"error": f"Unsupported export format '{fmt}'. Use 'docx' or 'pdf'."}), 400

    try:
        from demo_app.document_exporter import DocumentExporter
        if fmt == "docx":
            out_bytes = DocumentExporter.export_docx(
                text=text, score=score, tier=tier, mode=mode, title=title
            )
            mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            out_bytes = DocumentExporter.export_pdf(
                text=text, score=score, tier=tier, mode=mode, title=title
            )
            mimetype = "application/pdf"

        dl_name = DocumentExporter.safe_filename(base_name, fmt)
        return Response(
            out_bytes,
            mimetype=mimetype,
            headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return jsonify({"error": f"Export failed: {exc}"}), 500


@app.route("/api/repair", methods=["POST"])
def repair():
    data = request.get_json(force=True) or {}
    orig = data.get("original", "").strip()
    simp = data.get("simplification", "").strip()
    domain = data.get("domain", "auto").strip()
    target_lang = data.get("target_lang", "auto").strip().lower()

    if not orig or not simp:
        return jsonify({"error": "Please provide both the original clause and the simplified version."}), 400

    # Read API key strictly from the backend environment (.env or OS environment)
    backend_key = get_gemini_api_key()
    result = engine.auto_repair(orig, simp, domain=domain, api_key=backend_key, target_lang=target_lang)
    return jsonify(result)


@app.route("/api/gemini-status", methods=["GET"])
def gemini_status():
    backend_key = get_gemini_api_key()
    is_active = bool(backend_key)
    masked_key = ""
    if is_active:
        masked_key = backend_key[:6] + "..." + backend_key[-4:] if len(backend_key) > 10 else "***"
    return jsonify({
        "configured": is_active,
        "masked_key": masked_key,
        "privacy_masking": is_privacy_masking_enabled(),
        "model": "gemini-flash-latest / gemini-pro-latest",
        "description": "Google Gemini Pro Generative Plain-Language Author with Local Privacy Shield (Backend Configured)"
    })


@app.route("/api/simplify-levels", methods=["POST"])
def simplify_levels():
    data = request.get_json(force=True) or {}
    orig = data.get("original", "").strip()
    domain = data.get("domain", "auto").strip()

    if not orig:
        return jsonify({"error": "Please provide the original clause."}), 400

    result = engine.multilevel_simplify(orig, domain=domain)
    return jsonify(result)



@app.route("/api/examples")
def get_examples():
    return jsonify(engine.get_benchmark_examples())


@app.route("/api/dataset")
def get_dataset():
    query = request.args.get("q", "")
    lang = request.args.get("lang", "all")
    limit = int(request.args.get("limit", 45))
    results = engine.search_dataset(query=query, limit=limit, lang=lang)
    return jsonify(results)


@app.route("/api/stats")
def get_stats():
    total_annotations = len(engine.dataset_items)
    fr_count = sum(1 for it in engine.dataset_items if it.get("lang") == "fr")
    en_count = sum(1 for it in engine.dataset_items if it.get("lang") == "en")
    hi_count = sum(1 for it in engine.dataset_items if it.get("lang") == "hi")
    return jsonify({
        "total_annotations": total_annotations,
        "fr_count": fr_count,
        "en_count": en_count,
        "hi_count": hi_count,
        "categories_count": len(CATEGORIES),
        "source": "Multilingual Corpus (FrJUDGE + ContractNLI + MILDSum & Hindi)",
        "model": "JUDGEBERT (Calibrated Multilingual XLM-R + mDeBERTa-v3)",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting JUDGEBERT Interactive Demo on http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
