#!/usr/bin/env python3
"""
Flask web UI for the Gene Sequence Classification project.
Usage: pip install flask && python app.py
"""

import os
import io
from Bio import SeqIO
import numpy as np
from collections import Counter
from flask import Flask, render_template, request, jsonify

from gene_classifier import (
    load_data, preprocess,
)
from validation_pipeline import audit_dataset, run_experiment
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
_state = {}


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "Upload exceeds the 16 MB limit."}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audit", methods=["POST"])
def audit():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON settings object."}), 400
    try:
        sequences, labels, filenames, genes = load_data(data.get("dataset", "Dataset/Dataset"))
        if not sequences:
            raise ValueError("No sequences found at the selected dataset path.")
        return jsonify(audit_dataset(sequences, labels, genes, filenames))
    except (ValueError, TypeError, OSError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON settings object."}), 400
    try:
        kmer_size = int(data.get("kmer", 4))
        test_size = float(data.get("test_size", 0.3))
    except (ValueError, TypeError):
        return jsonify({"error": "K-mer size and test fraction must be numeric."}), 400
    model_choice = data.get("model", "rf")
    if model_choice not in ("rf", "svm", "lr", "gb", "all"):
        return jsonify({"error": "Unsupported model."}), 400
    use_smote = data.get("smote", False)
    dataset_path = data.get("dataset", "Dataset/Dataset")

    sequences, labels, filenames, gene_names = load_data(dataset_path)
    if not sequences:
        return jsonify({"error": f"No sequences found at '{dataset_path}'."}), 400

    label_counts = Counter(labels)
    gene_counts = Counter(gene_names)
    lengths = [len(s) for s in sequences]
    dataset_info = {
        "total": len(sequences),
        "healthy": label_counts.get(0, 0),
        "unhealthy": label_counts.get(1, 0),
        "genes": dict(sorted(gene_counts.items())),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": int(np.mean(lengths)),
    }

    try:
        result = run_experiment(
            sequences, labels, gene_names, k=kmer_size, model_name=model_choice,
            test_size=test_size, smote=use_smote,
            compare_features=bool(data.get("compare_features", False)), filenames=filenames,
        )
    except (ValueError, ImportError) as error:
        return jsonify({"error": str(error)}), 400

    pipeline = result["pipeline"]
    _state["pipeline"] = pipeline
    y_test = result["test_labels"]

    roc_data = None
    pr_data = None
    if result["probabilities"] is not None:
        y_proba = result["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data = {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_data = {
            "precision": [round(float(x), 4) for x in prec],
            "recall": [round(float(x), 4) for x in rec],
            "ap": round(float(average_precision_score(y_test, y_proba)), 4),
        }

    return jsonify({
        "dataset": dataset_info,
        "metrics": result["metrics"],
        "best_model": result["best_model"],
        "best_features": result["best_features"],
        "audit": result["audit"],
        "confusion_matrix": result["confusion_matrix"],
        "roc": roc_data,
        "pr": pr_data,
        "comparison": result["comparison"],
        "top_features": result["top_features"],
        "feature_shape": result["feature_shape"],
        "cv": result["cv"],
        "split": result["split"],
        "per_gene": result["per_gene"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    pipeline = _state.get("pipeline")
    if pipeline is None:
        return jsonify({"error": "No model trained yet. Train first."}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    gene = request.form.get("gene", "")
    references = pipeline.named_steps["features"].references_
    if gene not in references:
        return jsonify({"error": "Select a supported trained gene: APP, PSEN1 or PSEN2."}), 400
    try:
        content = file.read().decode("utf-8-sig").strip()
        if not content.startswith(">"):
            raise ValueError("FASTA must begin with a > header.")
        records = list(SeqIO.parse(io.StringIO(content), "fasta"))
        if not records or len(records) > 100:
            raise ValueError("Upload between 1 and 100 FASTA records from the selected gene.")
        seqs = [str(record.seq).upper() for record in records]
        headers = [record.description for record in records]
        reference_length = len(references[gene])
        for sequence in seqs:
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError("Every record must contain only A, C, G and T; ambiguous bases are not supported.")
            if not 0.95 * reference_length <= len(sequence) <= 1.05 * reference_length:
                raise ValueError(f"Expected a full-length {gene} sequence near {reference_length:,} bp (+/-5%). Fragments are not supported.")
        features = np.array([(sequence, gene) for sequence in seqs], dtype=object)
        preds = pipeline.predict(features)
        probas = pipeline.predict_proba(features)
    except (ValueError, UnicodeError) as error:
        return jsonify({"error": str(error)}), 400

    results = []
    for i, (header, pred) in enumerate(zip(headers, preds)):
        row = {
            "header": header[:80],
            "length": len(seqs[i]),
            "prediction": "Healthy" if pred == 0 else "Unhealthy",
            "class": int(pred),
            "gene": gene,
        }
        if probas is not None:
            row["confidence"] = round(float(max(probas[i])), 4)
        results.append(row)

    return jsonify({
        "predictions": results,
        "total": len(results),
        "healthy_count": sum(1 for r in results if r["class"] == 0),
        "unhealthy_count": sum(1 for r in results if r["class"] == 1),
        "warning": "Research prediction, not a diagnosis. Gene identity, coordinate alignment and clinical validity are not verified. Model probabilities are not calibrated disease risks.",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
