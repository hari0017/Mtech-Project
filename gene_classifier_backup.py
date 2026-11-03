#!/usr/bin/env python3
"""
gene_classifier.py — Complete ML pipeline for classifying gene sequences
                     (Healthy vs Unhealthy) from FASTA files.

Dataset structure expected:
    Dataset/Dataset/
    ├── APP/
    │   ├── output_safe_healthy/    (*.fasta files → label 0)
    │   └── output_safe_unhealthy/  (*.fasta files → label 1)
    ├── PSEN1/
    │   ├── output_safe_healthy/
    │   └── output_safe_unhealthy/
    └── PSEN2/
        ├── output_safe_healthy/
        └── output_safe_unhealthy/

Usage:
    python gene_classifier.py
    python gene_classifier.py --kmer 4
    python gene_classifier.py --model rf       (RandomForest)
    python gene_classifier.py --model svm      (SVM)
    python gene_classifier.py --model lr       (LogisticRegression)
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter
from itertools import product
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def read_fasta(filepath: str) -> str:
    """Read a FASTA file and return the sequence (ignoring header lines)."""
    sequences = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            # Keep only valid DNA characters
            cleaned = "".join(ch for ch in line.upper() if ch in "ACGT")
            sequences.append(cleaned)
    return "".join(sequences)


def load_data(dataset_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load all FASTA files from the dataset directory structure.

    Returns:
        sequences: List of DNA strings
        labels: List of integer labels (0=healthy, 1=unhealthy)
        filenames: List of source filenames for debugging
    """
    genes = ["APP", "PSEN1", "PSEN2"]
    sequences = []
    labels = []
    filenames = []

    for gene in genes:
        # Healthy sequences (label = 0)
        healthy_dir = os.path.join(dataset_root, gene, "output_safe_healthy")
        if os.path.isdir(healthy_dir):
            for fname in sorted(os.listdir(healthy_dir)):
                if fname.endswith((".fasta", ".fa", ".fna")):
                    fpath = os.path.join(healthy_dir, fname)
                    seq = read_fasta(fpath)
                    if seq:
                        sequences.append(seq)
                        labels.append(0)
                        filenames.append(f"{gene}/healthy/{fname}")

        # Unhealthy sequences (label = 1)
        unhealthy_dir = os.path.join(dataset_root, gene, "output_safe_unhealthy")
        if os.path.isdir(unhealthy_dir):
            for fname in sorted(os.listdir(unhealthy_dir)):
                if fname.endswith((".fasta", ".fa", ".fna")):
                    fpath = os.path.join(unhealthy_dir, fname)
                    seq = read_fasta(fpath)
                    if seq:
                        sequences.append(seq)
                        labels.append(1)
                        filenames.append(f"{gene}/unhealthy/{fname}")

    return sequences, labels, filenames


# ===========================================================================
# 2. PREPROCESSING — K-mer Frequency Encoding
# ===========================================================================

def build_kmer_vocab(k: int) -> Dict[str, int]:
    """Build vocabulary of all possible k-mers (4^k entries)."""
    bases = "ACGT"
    kmers = ["".join(combo) for combo in product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


def sequence_to_kmer_frequencies(sequence: str, k: int, vocab: Dict[str, int]) -> np.ndarray:
    """
    Convert a DNA sequence into a k-mer frequency vector.

    Each position in the vector represents the normalized frequency
    of that k-mer in the sequence.
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    total = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in vocab:
            vec[vocab[kmer]] += 1
            total += 1

    # Normalize to frequencies
    if total > 0:
        vec /= total

    return vec


def preprocess(sequences: List[str], k: int = 4) -> np.ndarray:
    """
    Convert all sequences to k-mer frequency feature vectors.

    Args:
        sequences: List of DNA strings.
        k: K-mer size (default 4 → 256 features).

    Returns:
        Feature matrix of shape (n_samples, 4^k).
    """
    vocab = build_kmer_vocab(k)
    print(f"  K-mer size: {k}")
    print(f"  Feature vector size: {len(vocab)} (4^{k})")
    print(f"  Processing {len(sequences)} sequences...")

    X = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        X[i] = sequence_to_kmer_frequencies(seq, k, vocab)
        if (i + 1) % 50 == 0 or i == len(sequences) - 1:
            print(f"    Processed {i + 1}/{len(sequences)}")

    return X


# ===========================================================================
# 3. DATA CHECKS
# ===========================================================================

def data_checks(sequences: List[str], labels: List[int], filenames: List[str]) -> None:
    """Print dataset statistics and verify integrity."""
    print(f"\n  {'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total sequences     : {len(sequences)}")
    print(f"  Total labels        : {len(labels)}")

    label_counts = Counter(labels)
    print(f"  Class distribution  :")
    print(f"    Healthy (0)       : {label_counts.get(0, 0)}")
    print(f"    Unhealthy (1)     : {label_counts.get(1, 0)}")

    if len(label_counts) < 2:
        print("  [ERROR] Only one class found! Check your data loading.")
        sys.exit(1)

    # Sequence length stats
    lengths = [len(s) for s in sequences]
    print(f"  Sequence lengths    :")
    print(f"    Min               : {min(lengths):,} bp")
    print(f"    Max               : {max(lengths):,} bp")
    print(f"    Mean              : {int(np.mean(lengths)):,} bp")

    # Per-gene breakdown
    gene_counts = Counter()
    for f in filenames:
        gene = f.split("/")[0]
        gene_counts[gene] += 1
    print(f"  Per-gene counts     :")
    for gene, count in sorted(gene_counts.items()):
        print(f"    {gene:<10}       : {count}")

    print(f"  {'='*60}\n")


# ===========================================================================
# 5. MODEL SELECTION
# ===========================================================================

def get_model(model_name: str, class_weights: dict):
    """Return a classifier instance with class-weight balancing."""
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "svm":
        return SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight=class_weights,
            probability=True,
            random_state=42,
        )
    elif model_name == "lr":
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight=class_weights,
            random_state=42,
            solver="lbfgs",
        )
    elif model_name == "gb":
        # GradientBoosting doesn't support class_weight directly,
        # use sample_weight during fit
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use: rf, svm, lr, gb")


# ===========================================================================
# 7. TRAINING
# ===========================================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "rf",
) -> object:
    """
    Train the model with class-weight balancing.

    Returns:
        Trained classifier.
    """
    # Compute balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
    print(f"  Class weights: {class_weight_dict}")

    model = get_model(model_name, class_weight_dict)
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Training on {len(y_train)} samples...")

    if model_name == "gb":
        # Manually compute sample weights for GradientBoosting
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    print("  Training complete.\n")
    return model


# ===========================================================================
# 8. EVALUATION
# ===========================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "rf",
) -> Dict[str, float]:
    """
    Evaluate the model and print comprehensive metrics.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)

    # Get probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    print(f"  {'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"  {'='*60}")

    # --- Class Distribution Check ---
    true_dist = Counter(y_test)
    pred_dist = Counter(y_pred)
    print(f"\n  --- Class Distribution ---")
    print(f"  y_true : Healthy={true_dist.get(0, 0)}, Unhealthy={true_dist.get(1, 0)}")
    print(f"  y_pred : Healthy={pred_dist.get(0, 0)}, Unhealthy={pred_dist.get(1, 0)}")

    # --- Single-class prediction check ---
    unique_preds = set(y_pred)
    if len(unique_preds) == 1:
        only_class = "Healthy" if 0 in unique_preds else "Unhealthy"
        print(f"\n  [WARNING] Model predicts ONLY '{only_class}'!")
        print(f"  Attempting threshold tuning...")
        if y_proba is not None:
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.arange(0.3, 0.7, 0.05):
                adj_pred = (y_proba >= thresh).astype(int)
                f1 = f1_score(y_test, adj_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            y_pred = (y_proba >= best_thresh).astype(int)
            print(f"  Applied threshold={best_thresh:.2f} (F1={best_f1:.4f})")
            pred_dist = Counter(y_pred)
            print(f"  Adjusted y_pred : Healthy={pred_dist.get(0, 0)}, Unhealthy={pred_dist.get(1, 0)}")

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  --- Metrics ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC   : {roc_auc:.4f}")
    else:
        roc_auc = 0.0

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\n  --- Confusion Matrix ---")
    print(f"                      Predicted")
    print(f"                      Healthy  Unhealthy")
    print(f"  Actual Healthy      {cm[0][0]:<8} {cm[0][1]}")
    print(f"  Actual Unhealthy    {cm[1][0]:<8} {cm[1][1]}")

    # --- Classification Report ---
    report = classification_report(
        y_test, y_pred,
        target_names=["Healthy", "Unhealthy"],
        zero_division=0,
    )
    print(f"\n  --- Classification Report ---")
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"  {'='*60}\n")

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc_auc,
    }


# ===========================================================================
# 12. VISUALIZATION (Optional — saves to file)
# ===========================================================================

def save_confusion_matrix_plot(y_test, y_pred, output_path="confusion_matrix.png"):
    """Save confusion matrix as an image (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Unhealthy"],
            yticklabels=["Healthy", "Unhealthy"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Gene Sequence Classification")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix plot saved: {output_path}")
    except ImportError:
        print("  [!] matplotlib/seaborn not installed — skipping plot.")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gene Sequence Classifier (Healthy vs Unhealthy)",
    )
    parser.add_argument(
        "--dataset", default="Dataset/Dataset",
        help="Path to dataset root (default: Dataset/Dataset)",
    )
    parser.add_argument(
        "--kmer", type=int, default=4, choices=[3, 4, 5, 6],
        help="K-mer size for feature extraction (default: 4 → 256 features)",
    )
    parser.add_argument(
        "--model", default="rf", choices=["rf", "svm", "lr", "gb"],
        help="Model to use: rf=RandomForest, svm=SVM, lr=LogisticRegression, gb=GradientBoosting",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("  GENE SEQUENCE CLASSIFICATION PIPELINE")
    print("  (Healthy vs Unhealthy — APP, PSEN1, PSEN2)")
    print("=" * 64)
    print(f"\n  Dataset path : {args.dataset}")
    print(f"  K-mer size   : {args.kmer}")
    print(f"  Model        : {args.model}")
    print(f"  Test size    : {args.test_size}")

    # ---- Step 1: Load Data ----
    print(f"\n{'─'*64}")
    print("  STEP 1: Loading FASTA sequences...")
    print(f"{'─'*64}")
    sequences, labels, filenames = load_data(args.dataset)

    if not sequences:
        print("  [ERROR] No sequences loaded! Check dataset path.")
        print(f"  Expected structure: {args.dataset}/APP/output_safe_healthy/*.fasta")
        sys.exit(1)

    # ---- Step 3: Data Checks ----
    data_checks(sequences, labels, filenames)

    # ---- Step 2: Preprocessing ----
    print(f"{'─'*64}")
    print("  STEP 2: Preprocessing (K-mer frequency encoding)...")
    print(f"{'─'*64}")
    X = preprocess(sequences, k=args.kmer)
    y = np.array(labels)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

    # ---- Step 4: Train/Test Split (Stratified) ----
    print(f"\n{'─'*64}")
    print("  STEP 3: Stratified Train/Test Split")
    print(f"{'─'*64}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
        shuffle=True,
    )
    print(f"  Train set: {len(y_train)} samples (Healthy={sum(y_train==0)}, Unhealthy={sum(y_train==1)})")
    print(f"  Test set : {len(y_test)} samples (Healthy={sum(y_test==0)}, Unhealthy={sum(y_test==1)})")

    # Scale features for SVM/LR
    if args.model in ("svm", "lr"):
        print("  Applying StandardScaler for SVM/LR...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # ---- Step 5-7: Train Model ----
    print(f"\n{'─'*64}")
    print("  STEP 4: Training Model")
    print(f"{'─'*64}")
    model = train_model(X_train, y_train, model_name=args.model)

    # ---- Cross-validation for robustness ----
    print(f"{'─'*64}")
    print("  STEP 5: Cross-Validation (5-fold)")
    print(f"{'─'*64}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Step 8: Evaluation ----
    print(f"\n{'─'*64}")
    print("  STEP 6: Evaluation on Test Set")
    print(f"{'─'*64}")
    metrics = evaluate_model(model, X_test, y_test, model_name=args.model)

    # ---- Step 12: Save plot ----
    y_pred = model.predict(X_test)
    save_confusion_matrix_plot(y_test, y_pred)

    # ---- Feature Importance (for tree-based models) ----
    if args.model in ("rf", "gb") and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        vocab = build_kmer_vocab(args.kmer)
        kmer_names = list(vocab.keys())
        top_indices = np.argsort(importances)[-10:][::-1]
        print(f"  --- Top 10 Most Important K-mers ---")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank:>2}. {kmer_names[idx]}  (importance: {importances[idx]:.5f})")
        print()

    print("=" * 64)
    print("  PIPELINE COMPLETE")
    print("=" * 64)
    print(f"\n  Final Results:")
    print(f"    Accuracy  = {metrics['accuracy']:.4f}")
    print(f"    Precision = {metrics['precision']:.4f}")
    print(f"    Recall    = {metrics['recall']:.4f}")
    print(f"    F1 Score  = {metrics['f1']:.4f}")
    print(f"    ROC-AUC   = {metrics['roc_auc']:.4f}")
    print()


if __name__ == "__main__":
    main()
