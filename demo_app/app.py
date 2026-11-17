import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add parent directory to path so relative imports work
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from demo_app.model_engine import JudgeBertEngine, CATEGORIES

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

    if not orig or not simp:
        return jsonify({"error": "Please provide both the original clause and the simplified version."}), 400

    result = engine.evaluate_pair(orig, simp, domain=domain)
    return jsonify(result)


@app.route("/api/repair", methods=["POST"])
def repair():
    data = request.get_json(force=True) or {}
    orig = data.get("original", "").strip()
    simp = data.get("simplification", "").strip()
    domain = data.get("domain", "auto").strip()

    if not orig or not simp:
        return jsonify({"error": "Please provide both the original clause and the simplified version."}), 400

    result = engine.auto_repair(orig, simp, domain=domain)
    return jsonify(result)


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
    limit = int(request.args.get("limit", 40))
    results = engine.search_dataset(query=query, limit=limit)
    return jsonify(results)


@app.route("/api/stats")
def get_stats():
    total_annotations = len(engine.dataset_items)
    return jsonify({
        "total_annotations": total_annotations,
        "categories_count": len(CATEGORIES),
        "source": "FrJUDGE (EMNLP 2025)",
        "model": "JUDGEBERT (Calibrated on FrJUDGE & MiniLM/CamemBERT)",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting JUDGEBERT Interactive Demo on http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
