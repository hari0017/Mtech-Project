#!/usr/bin/env python3
"""
main.py — End-to-end pipeline for converting a pangenome graph to DNABERT
input and running variant pathogenicity prediction.

Pipeline stages:
  1. Parse a GFA/rGFA pangenome graph.
  2. Build a directed graph and convert to DAG (SCC contraction).
  3. Extract linear paths (topological + random walk).
  4. Linearize paths into DNA sequences.
  5. Tokenize sequences into k-mers.
  6. Encode tokens with special tokens, positional IDs, padding.
  7. (Optional) Load labels from CSV/VCF annotation file.
  8. (Optional) Fine-tune DNABERT on labelled data.
  9. Run inference with DNABERT.
    10. Evaluate predictions on held-out labelled inputs.

Usage:
    python main.py --model trained_model                    # inference only, no scores without labels
    python main.py --gfa my_graph.gfa                       # custom GFA file
    python main.py --labels labels.csv                      # use real labels for evaluation
    python main.py --labels labels.csv --train              # fine-tune then evaluate
    python main.py --labels data.vcf --train --epochs 20    # train with VCF annotations
    python main.py --model zhihan1996/DNA_bert_6 --kmer 6 --train --labels labels.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import List

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import set_seed

from utils import PipelineConfig, print_section, setup_logger
from graph_processing import process_graph
from path_extraction import extract_paths
from sequence_processing import linearize_paths, encode_tokens
from tokenizer_module import KmerTokenizer
from annotation_loader import load_labels, get_labels_for_sequences
from dnabert_model import (
    DNABertClassifier,
    PredictionResult,
    evaluate_predictions,
    predict_with_threshold,
    LABEL_MAP,
)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    """Build a :class:`PipelineConfig` from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pangenome -> DNABERT pathogenicity prediction pipeline",
    )
    parser.add_argument(
        "--gfa", default="sample.gfa",
        help="Path to the input GFA / rGFA file (default: sample.gfa)",
    )
    parser.add_argument(
        "--kmer", type=int, default=3, choices=[3, 4, 5, 6],
        help="K-mer size for tokenization (default: 3)",
    )
    parser.add_argument(
        "--max-len", type=int, default=128,
        help="Maximum encoded sequence length including special tokens (default: 128)",
    )
    parser.add_argument(
        "--max-paths", type=int, default=10,
        help="Max topological paths to extract (default: 10)",
    )
    parser.add_argument(
        "--rw-paths", type=int, default=3,
        help="Number of random-walk paths to sample (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model", default="zhihan1996/DNA_bert_3",
        help="HuggingFace model name or local path (default: zhihan1996/DNA_bert_3)",
    )
    parser.add_argument(
        "--labels", default="",
        help="Path to a CSV/TSV/VCF file with real labels (columns: sequence_id, label)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Fine-tune the model on labelled data before inference",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate for fine-tuning (default: 2e-5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation fraction of non-test groups (default: 0.2)",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.3,
        help="Held-out test fraction of distinct encoded-input groups (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir", default="trained_model",
        help="Directory to save the fine-tuned model (default: trained_model)",
    )

    args = parser.parse_args()
    return PipelineConfig(
        gfa_path=args.gfa,
        kmer_size=args.kmer,
        max_seq_length=args.max_len,
        max_paths=args.max_paths,
        random_walk_paths=args.rw_paths,
        random_seed=args.seed,
        model_name=args.model,
        labels_file=args.labels,
        train=args.train,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        output_dir=args.output_dir,
        test_split=args.test_split,
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_graph_processing(cfg: PipelineConfig):
    """Stage 1-2: Parse GFA and build DAG."""
    print_section("STAGE 1-2: Graph Processing (GFA -> DAG)")
    dag, gfa, scc_map = process_graph(cfg.gfa_path)

    print(f"  Segments in GFA   : {len(gfa.segments)}")
    print(f"  Links in GFA      : {len(gfa.links)}")
    print(f"  Named paths (P)   : {len(gfa.paths)}")
    print(f"  DAG nodes         : {dag.number_of_nodes()}")
    print(f"  DAG edges         : {dag.number_of_edges()}")
    print(f"  SCCs contracted   : {sum(1 for v in scc_map.values() if len(v) > 1)}")

    return dag, gfa, scc_map


def stage_path_extraction(cfg: PipelineConfig, dag):
    """Stage 3: Extract linear paths from the DAG."""
    print_section("STAGE 3: Path Extraction")
    paths = extract_paths(
        dag,
        max_paths=cfg.max_paths,
        random_walk_paths=cfg.random_walk_paths,
        random_seed=cfg.random_seed,
    )

    for i, p in enumerate(paths):
        print(f"  Path {i + 1:>2}: {' -> '.join(p)}")

    if not paths:
        print("  [!] No paths extracted -- check your GFA file.")

    return paths


def stage_linearization(dag, paths: List[List[str]]):
    """Stage 4: Concatenate DNA sequences along each path."""
    print_section("STAGE 4: Sequence Linearization")
    sequences = linearize_paths(dag, paths)

    for i, seq in enumerate(sequences):
        display = seq[:60] + ("..." if len(seq) > 60 else "")
        print(f"  Seq {i + 1:>2} ({len(seq):>6} bp): {display}")

    return sequences


def stage_tokenization(cfg: PipelineConfig, sequences: List[str]):
    """Stage 5: K-mer tokenization."""
    print_section(f"STAGE 5: Tokenization (k={cfg.kmer_size})")
    tokenizer = KmerTokenizer.from_pretrained(cfg.model_name, k=cfg.kmer_size)

    all_tokens: List[List[str]] = []
    for i, seq in enumerate(sequences):
        tokens = tokenizer.tokenize(seq)
        all_tokens.append(tokens)
        preview = " ".join(tokens[:12])
        suffix = " ..." if len(tokens) > 12 else ""
        print(f"  Seq {i + 1:>2}: {len(tokens):>5} tokens  [{preview}{suffix}]")

    print(f"\n  Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer, all_tokens


def stage_encoding(cfg: PipelineConfig, tokenizer: KmerTokenizer, all_tokens, sequences):
    """Stage 6: Add special tokens, positional IDs, pad/truncate."""
    print_section(f"STAGE 6: Sequence Encoding (max_length={cfg.max_seq_length})")
    vocab = tokenizer.get_vocab()
    if cfg.max_seq_length < 3:
        raise ValueError("--max-len must leave room for special tokens and at least one k-mer.")

    encoded_list = []
    for i, tokens in enumerate(all_tokens):
        if not tokens:
            raise ValueError(f"Sequence {i + 1} has no k-mers; check its length and --kmer.")
        if len(tokens) > cfg.max_seq_length - 2:
            print(f"  WARNING: Sequence {i + 1} truncated from {len(tokens)} to "
                  f"{cfg.max_seq_length - 2} k-mers; variants beyond this window are not represented.")
        enc = encode_tokens(
            tokens,
            vocab=vocab,
            max_length=cfg.max_seq_length,
            cls_token=cfg.cls_token,
            sep_token=cfg.sep_token,
            pad_token=cfg.pad_token,
        )
        enc.original_sequence = sequences[i]
        encoded_list.append(enc)

        real_tokens = sum(enc.attention_mask)
        print(
            f"  Seq {i + 1:>2}: "
            f"total={len(enc.token_ids)}, "
            f"real={real_tokens}, "
            f"padded={len(enc.token_ids) - real_tokens}"
        )

    return encoded_list


def stage_prediction(cfg: PipelineConfig, tokenizer: KmerTokenizer, encoded_list, threshold=0.5):
    """Stage 7-8: Load model and run inference."""
    print_section("STAGE 7-8: DNABERT Model & Prediction")

    classifier = DNABertClassifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        vocab_size=tokenizer.vocab_size,
        max_seq_length=cfg.max_seq_length,
    )

    token_ids_batch = [e.token_ids for e in encoded_list]
    attention_mask_batch = [e.attention_mask for e in encoded_list]

    results = classifier.predict(token_ids_batch, attention_mask_batch)
    for result, predicted in zip(results, predict_with_threshold(results, threshold)):
        result.predicted_class = predicted
        result.predicted_label = LABEL_MAP[predicted]

    for r in results:
        prob_str = ", ".join(f"{p:.4f}" for p in r.probabilities)
        print(
            f"  Seq {r.sequence_index + 1:>2}  ->  "
            f"{r.predicted_label:<12}  "
            f"(probs: [{prob_str}])"
        )

    return classifier, results


def stage_load_labels(cfg: PipelineConfig, num_sequences: int):
    """Load real labels from a CSV/TSV/VCF annotation file."""
    print_section("STAGE: Loading Annotations")

    label_map = load_labels(cfg.labels_file)
    indices, labels = get_labels_for_sequences(label_map, num_sequences)

    benign_count = labels.count(0)
    pathogenic_count = labels.count(1)
    print(f"  Label file     : {cfg.labels_file}")
    print(f"  Labels loaded  : {len(labels)}")
    print(f"  BENIGN         : {benign_count}")
    print(f"  PATHOGENIC     : {pathogenic_count}")
    print(f"  Unlabelled     : {num_sequences - len(labels)}")

    return label_map, indices, labels


def stage_training(
    cfg: PipelineConfig,
    classifier: DNABertClassifier,
    encoded_list,
    indices: List[int],
    labels: List[int],
):
    """Fine-tune the model on labelled data."""
    print_section("STAGE: Fine-tuning DNABERT")

    token_ids_batch = [encoded_list[i].token_ids for i in indices]
    attention_mask_batch = [encoded_list[i].attention_mask for i in indices]

    print(f"  Training samples : {len(labels)}")
    print(f"  Epochs           : {cfg.epochs}")
    print(f"  Learning rate    : {cfg.learning_rate}")
    print(f"  Batch size       : {cfg.batch_size}")
    print("  Validation       : separate partition for threshold selection")
    print()

    history = classifier.train_model(
        token_ids_batch=token_ids_batch,
        attention_mask_batch=attention_mask_batch,
        labels=labels,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        val_split=0.0,
    )

    # Save model
    classifier.save_model(cfg.output_dir)
    print(f"\n  Model saved to   : {cfg.output_dir}/")

    return history


def split_labelled_inputs(cfg, encoded_list, indices, labels):
    """Stratify unique model inputs and keep identical inputs together."""
    if not 0 < cfg.test_split < 1 or not 0 < cfg.val_split < 1:
        raise ValueError("Test and validation fractions must be between 0 and 1.")
    if len(indices) != len(labels) or len(set(indices)) != len(indices):
        raise ValueError("Labels must map one-to-one to unique sequence indices.")
    if set(labels) != {0, 1}:
        raise ValueError("Both BENIGN (0) and PATHOGENIC (1) labels are required.")
    groups = {}
    for index, label in zip(indices, labels):
        if not 0 <= index < len(encoded_list):
            raise ValueError("Label index is outside the encoded sequence collection.")
        key = tuple(encoded_list[index].token_ids)
        if key in groups and groups[key]["label"] != label:
            raise ValueError(
                "Conflicting labels for identical encoded inputs. Check annotations "
                "and sequence truncation before training."
            )
        groups.setdefault(key, {"label": label, "indices": []})["indices"].append(index)
    grouped = list(groups.values())
    group_ids = list(range(len(grouped)))
    try:
        remaining, test = train_test_split(
            group_ids, test_size=cfg.test_split, random_state=cfg.random_seed,
            stratify=[group["label"] for group in grouped],
        )
        train, validation = train_test_split(
            remaining, test_size=cfg.val_split, random_state=cfg.random_seed,
            stratify=[grouped[group_id]["label"] for group_id in remaining],
        )
        partitions = {"train": train, "validation": validation, "test": test}
        if any({grouped[group_id]["label"] for group_id in partition} != {0, 1}
               for partition in partitions.values()):
            raise ValueError("A partition is missing a class.")
    except ValueError as exc:
        raise ValueError(
            "Not enough distinct labelled inputs for stratified train/validation/test "
            "splits with both classes. Add verified data or adjust split fractions."
        ) from exc
    return {
        name: sorted(index for group_id in partition for index in grouped[group_id]["indices"])
        for name, partition in partitions.items()
    }


def predict_indices(classifier, encoded_list, indices):
    results = classifier.predict(
        [encoded_list[index].token_ids for index in indices],
        [encoded_list[index].attention_mask for index in indices],
    )
    for result, index in zip(results, indices):
        result.sequence_index = index
    return results


def select_validation_threshold(results, label_map):
    """Select F1 threshold on validation data; retain 0.5 for ties."""
    labels = [label_map[result.sequence_index] for result in results]
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both classes in validation data.")
    candidates = [0.5] + [step / 20 for step in range(1, 20) if step != 10]
    return max(candidates, key=lambda threshold: f1_score(
        labels, predict_with_threshold(results, threshold), zero_division=0,
    ))


def run_training_experiment(cfg, classifier, encoded_list, label_map, partitions):
    train_indices = partitions["train"]
    history = stage_training(
        cfg, classifier, encoded_list, train_indices,
        [label_map[index] for index in train_indices],
    )
    validation_results = predict_indices(classifier, encoded_list, partitions["validation"])
    threshold = select_validation_threshold(validation_results, label_map)
    print(f"  Validation-selected threshold: {threshold:.2f}")
    test_results = predict_indices(classifier, encoded_list, partitions["test"])
    print_section("HELD-OUT TEST EVALUATION")
    metrics = stage_evaluation(test_results, label_map, True, threshold)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": {package: version(package) for package in
                             ("torch", "transformers", "numpy", "scikit-learn")},
        "config": cfg.to_dict(),
        "split_indices": partitions,
        "class_counts": {
            name: {str(label): sum(label_map[index] == label for index in partition)
                   for label in (0, 1)}
            for name, partition in partitions.items()
        },
        "input_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("gfa", cfg.gfa_path), ("labels", cfg.labels_file))
        },
        "threshold": threshold,
        "threshold_source": "validation",
        "evaluation_source": "held_out_test",
        "history": history,
        "metrics": metrics,
        "limitations": [
            "Exact encoded-input grouping does not exclude related biological samples.",
            "Annotation provenance and path-to-label correspondence require verification.",
            "Research use only; not clinically validated.",
        ],
    }
    output_path = Path(cfg.output_dir) / "experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(f"  Experiment record: {output_path}")
    return record


def stage_evaluation(
    results: List[PredictionResult],
    label_map=None,
    has_real_labels: bool = False,
    threshold: float = 0.5,
):
    """Evaluate labelled predictions at a previously fixed threshold."""
    if has_real_labels and label_map:
        print_section("STAGE: Evaluation (real labels)")

        true_labels = []
        predicted = []
        labelled_results = []
        for r in results:
            if r.sequence_index in label_map:
                true_labels.append(label_map[r.sequence_index])
                labelled_results.append(r)

        if not true_labels:
            print("  [!] No matching labels found for predicted sequences.")
            return {}

        predicted = predict_with_threshold(labelled_results, threshold=threshold)
        metrics = evaluate_predictions(true_labels, predicted)
        probabilities = [result.probabilities[1] for result in labelled_results]
        metrics["roc_auc"] = float(roc_auc_score(true_labels, probabilities)) if len(set(true_labels)) == 2 else None
        metrics["average_precision"] = float(average_precision_score(true_labels, probabilities)) if len(set(true_labels)) == 2 else None
        metrics["confusion_matrix"] = confusion_matrix(true_labels, predicted, labels=[0, 1]).tolist()
        metrics["samples"] = len(true_labels)
        metrics["threshold"] = threshold
        print(f"\n  Sequences evaluated : {len(true_labels)}")
        print(f"  Accuracy            : {metrics['accuracy']:.4f}")
        print(f"  F1 Score            : {metrics['f1']:.4f}")
        print(f"  Precision           : {metrics['precision']:.4f}")
        print(f"  Recall              : {metrics['recall']:.4f}")

        return metrics

    else:
        print_section("STAGE: Evaluation unavailable")
        print(
            "  Labelled data required. Use --labels <file> with verified annotations.\n"
            "  No accuracy, precision, recall or F1 score has been calculated."
        )
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full pangenome -> DNABERT pipeline."""
    cfg = parse_args()
    setup_logger()
    set_seed(cfg.random_seed)
    if cfg.train and not cfg.labels_file:
        raise ValueError("--train requires --labels with verified annotations.")

    print_section("PANGENOME -> DNABERT PATHOGENICITY PREDICTION PIPELINE")
    print(f"  GFA file       : {cfg.gfa_path}")
    print(f"  K-mer size     : {cfg.kmer_size}")
    print(f"  Max seq length : {cfg.max_seq_length}")
    print(f"  Model          : {cfg.model_name}")
    print(f"  Random seed    : {cfg.random_seed}")
    print(f"  Labels file    : {cfg.labels_file or '(none - evaluation unavailable)'}")
    print(f"  Training mode  : {'ON' if cfg.train else 'OFF'}")

    # Stage 1-2: Graph processing
    dag, gfa, scc_map = stage_graph_processing(cfg)
    if dag.number_of_nodes() == 0:
        print("\n[ERROR] DAG is empty after processing — nothing to do.")
        sys.exit(1)

    # Stage 3: Path extraction
    paths = stage_path_extraction(cfg, dag)
    if not paths:
        print("\n[ERROR] No paths could be extracted — check GFA structure.")
        sys.exit(1)

    # Stage 4: Linearization
    sequences = stage_linearization(dag, paths)
    if not sequences:
        print("\n[ERROR] No sequences generated — check segment data.")
        sys.exit(1)

    # Stage 5: Tokenization
    tokenizer, all_tokens = stage_tokenization(cfg, sequences)

    # Stage 6: Encoding
    encoded_list = stage_encoding(cfg, tokenizer, all_tokens, sequences)

    # Load labels (if provided)
    label_map = None
    indices = None
    labels = None
    has_real_labels = bool(cfg.labels_file)

    if has_real_labels:
        label_map, indices, labels = stage_load_labels(cfg, len(sequences))

    if cfg.train:
        partitions = split_labelled_inputs(cfg, encoded_list, indices, labels)
        classifier = DNABertClassifier(
            model_name=cfg.model_name, num_labels=cfg.num_labels,
            vocab_size=tokenizer.vocab_size, max_seq_length=cfg.max_seq_length,
        )
        run_training_experiment(cfg, classifier, encoded_list, label_map, partitions)
        tokenizer.save_pretrained(cfg.output_dir)
    else:
        threshold = 0.5
        record_path = Path(cfg.model_name) / "experiment.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record["config"]["kmer_size"] != cfg.kmer_size or
                    record["config"]["max_seq_length"] != cfg.max_seq_length):
                raise ValueError("Inference --kmer and --max-len must match the saved experiment.")
            threshold = record["threshold"]
        classifier, results = stage_prediction(cfg, tokenizer, encoded_list, threshold)
        if has_real_labels:
            print("  Evaluation independence depends on the supplied checkpoint and label provenance.")
        stage_evaluation(results, label_map=label_map, has_real_labels=has_real_labels,
                         threshold=threshold)

    print_section("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
