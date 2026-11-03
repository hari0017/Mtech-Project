#!/usr/bin/env python3
"""
gene_classifier.py — Improved ML pipeline for classifying gene sequences
                     (Healthy vs Unhealthy) from FASTA files.

Key improvements over baseline:
  - SMOTE oversampling for class imbalance
  - Multi-model comparison (RF, SVM, LR, GradientBoosting, XGBoost, LightGBM)
  - Stratified K-Fold cross-validation with multiple metrics
  - Hyperparameter tuning via RandomizedSearchCV
  - Feature scaling + optional feature selection (mutual information)
  - Additional genomic features (GC content, nucleotide fractions)
  - Precision-Recall curve, ROC curve, confusion matrix visualizations
  - Optimal threshold tuning for medical classification
  - Soft-voting ensemble option
  - Per-class metrics to detect single-class prediction bias

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
    python gene_classifier.py --model rf
    python gene_classifier.py --model all     (compare all models)
    python gene_classifier.py --smote         (enable SMOTE)
    python gene_classifier.py --tune          (hyperparameter tuning)
    python gene_classifier.py --ensemble      (soft-voting ensemble)
"""

import os
import sys
import argparse
import warnings
import numpy as np
from collections import Counter
from itertools import product
from typing import List, Tuple, Dict

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_validate, RandomizedSearchCV,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
            cleaned = "".join(ch for ch in line.upper() if ch in "ACGT")
            sequences.append(cleaned)
    return "".join(sequences)


def load_data(dataset_root: str) -> Tuple[List[str], List[int], List[str], List[str]]:
    """
    Load all FASTA files from the dataset directory structure.

    Returns:
        sequences: List of DNA strings
        labels:    List of integer labels (0=healthy, 1=unhealthy)
        filenames: List of source filenames for debugging
        gene_names: List of gene name per sequence
    """
    genes = ["APP", "PSEN1", "PSEN2"]
    sequences = []
    labels = []
    filenames = []
    gene_names = []

    for gene in genes:
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
                        gene_names.append(gene)

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
                        gene_names.append(gene)

    return sequences, labels, filenames, gene_names


# ===========================================================================
# 2. PREPROCESSING — K-mer Frequency + Genomic Features
# ===========================================================================

def build_kmer_vocab(k: int) -> Dict[str, int]:
    """Build vocabulary of all possible k-mers (4^k entries)."""
    bases = "ACGT"
    kmers = ["".join(combo) for combo in product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


def sequence_to_kmer_frequencies(sequence: str, k: int, vocab: Dict[str, int]) -> np.ndarray:
    """Convert a DNA sequence into a normalised k-mer frequency vector."""
    vec = np.zeros(len(vocab), dtype=np.float64)
    total = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in vocab:
            vec[vocab[kmer]] += 1
            total += 1
    if total > 0:
        vec /= total
    return vec


def compute_sequence_features(sequence: str) -> np.ndarray:
    """
    Compute additional hand-crafted genomic features:
      A%, C%, G%, T%, GC-content, AT/GC ratio
    """
    length = len(sequence)
    if length == 0:
        return np.zeros(6, dtype=np.float64)

    counts = Counter(sequence)
    a_frac = counts.get("A", 0) / length
    c_frac = counts.get("C", 0) / length
    g_frac = counts.get("G", 0) / length
    t_frac = counts.get("T", 0) / length

    gc_content = c_frac + g_frac
    at_gc_ratio = (a_frac + t_frac) / gc_content if gc_content > 0 else 0.0

    return np.array([a_frac, c_frac, g_frac, t_frac, gc_content, at_gc_ratio],
                    dtype=np.float64)


def build_gene_references(sequences: List[str], gene_names: List[str]) -> Dict[str, str]:
    """Build a reference sequence per gene using the first healthy sequence."""
    refs = {}
    for seq, gene in zip(sequences, gene_names):
        if gene not in refs:
            refs[gene] = seq
    return refs


def compute_mutation_features(sequence: str, reference: str, k: int = 4) -> np.ndarray:
    """
    Detect mutations relative to reference and extract discriminative features:
      - mutation_count: number of SNV differences
      - mutation_pos_mean: mean relative position of mutations
      - mutation_pos_std: std of relative mutation positions
      - mutation_pos_min: min relative position
      - mutation_pos_max: max relative position
      - n_transitions: count of transition mutations (A<>G, C<>T)
      - n_transversions: count of transversion mutations
      - local_gc_content: GC content in 50bp window around mutations
      - mutation_region: which region (0-4) the primary mutation falls in
      - context_entropy: Shannon entropy of 20bp context around mutation
      - local_kmer features: k-mer frequencies in context window (4^k values)
    """
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    min_len = min(len(sequence), len(reference))

    # Find mutation positions
    mut_positions = []
    mut_types = []  # 1=transition, 0=transversion
    for i in range(min_len):
        if sequence[i] != reference[i]:
            mut_positions.append(i)
            mut_types.append(1 if (reference[i], sequence[i]) in transitions else 0)

    # Length difference mutations
    len_diff = abs(len(sequence) - len(reference))

    n_muts = len(mut_positions)
    if n_muts == 0:
        # No mutations detected — return zeros
        n_context_kmers = 4 ** min(k, 4)
        return np.zeros(11 + n_context_kmers, dtype=np.float64)

    rel_positions = [p / max(len(reference), 1) for p in mut_positions]

    # Skip shared mutation (position 0 if present in many), focus on unique
    # Use the last mutation as the "primary" one (shared is usually first)
    primary_pos = mut_positions[-1]
    primary_rel = rel_positions[-1]

    # Local context analysis around primary mutation
    ctx_window = 50
    ctx_start = max(0, primary_pos - ctx_window)
    ctx_end = min(min_len, primary_pos + ctx_window + 1)
    context = reference[ctx_start:ctx_end]

    # Local GC content
    ctx_counts = Counter(context)
    ctx_len = len(context)
    local_gc = (ctx_counts.get("G", 0) + ctx_counts.get("C", 0)) / max(ctx_len, 1)

    # Context entropy
    import math
    entropy = 0.0
    for base in "ACGT":
        p = ctx_counts.get(base, 0) / max(ctx_len, 1)
        if p > 0:
            entropy -= p * math.log2(p)

    # Mutation region (divide gene into 5 regions)
    mutation_region = min(int(primary_rel * 5), 4)

    # Context k-mer frequencies (smaller k for context)
    ctx_k = min(k, 4)
    ctx_vocab = build_kmer_vocab(ctx_k)
    ctx_kmer_vec = sequence_to_kmer_frequencies(context, ctx_k, ctx_vocab)

    base_features = np.array([
        n_muts,                                          # mutation count
        np.mean(rel_positions),                          # mean relative position
        np.std(rel_positions) if n_muts > 1 else 0.0,   # std of positions
        min(rel_positions),                              # min position
        max(rel_positions),                              # max position
        sum(mut_types),                                  # transition count
        n_muts - sum(mut_types),                         # transversion count
        local_gc,                                        # local GC around mutation
        float(mutation_region),                          # gene region (0-4)
        entropy,                                         # context entropy
        float(len_diff),                                 # length difference
    ], dtype=np.float64)

    return np.concatenate([base_features, ctx_kmer_vec])


def preprocess(sequences: List[str], k: int = 4,
               gene_names: List[str] = None,
               references: Dict[str, str] = None) -> np.ndarray:
    """
    Convert all sequences to k-mer frequency + genomic + mutation features.

    Returns:
        Feature matrix of shape (n_samples, n_features).
    """
    vocab = build_kmer_vocab(k)
    n_kmer = len(vocab)
    n_genomic = 6

    # Build mutation features if gene_names provided
    use_mutation = gene_names is not None and len(gene_names) == len(sequences)
    n_ctx_kmers = 4 ** min(k, 4)
    n_mutation = 11 + n_ctx_kmers if use_mutation else 0
    n_gene_onehot = 3 if use_mutation else 0  # APP, PSEN1, PSEN2
    total_features = n_kmer + n_genomic + n_mutation + n_gene_onehot

    print(f"  K-mer size: {k}")
    print(f"  K-mer features: {n_kmer} (4^{k})")
    print(f"  Genomic features: {n_genomic} (A%, C%, G%, T%, GC%, AT/GC ratio)")
    if use_mutation:
        print(f"  Mutation features: {n_mutation} (position, type, context)")
        print(f"  Gene one-hot: {n_gene_onehot}")
    print(f"  Total features: {total_features}")
    print(f"  Processing {len(sequences)} sequences...")

    # Build references if using mutation features
    refs = {}
    gene_map = {"APP": 0, "PSEN1": 1, "PSEN2": 2}
    if use_mutation:
        refs = references if references is not None else build_gene_references(sequences, gene_names)
        missing = set(gene_names) - set(refs)
        if missing:
            raise ValueError(f"No training reference for genes: {sorted(missing)}")
        print(f"  References built for: {list(refs.keys())}")

    X = np.zeros((len(sequences), total_features), dtype=np.float64)
    for i, seq in enumerate(sequences):
        offset = 0
        X[i, offset:offset + n_kmer] = sequence_to_kmer_frequencies(seq, k, vocab)
        offset += n_kmer
        X[i, offset:offset + n_genomic] = compute_sequence_features(seq)
        offset += n_genomic

        if use_mutation:
            gene = gene_names[i]
            ref = refs[gene]
            mut_feats = compute_mutation_features(seq, ref, k)
            X[i, offset:offset + n_mutation] = mut_feats
            offset += n_mutation

            # Gene one-hot encoding
            gene_idx = gene_map.get(gene, 0)
            X[i, offset + gene_idx] = 1.0

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

    label_counts = Counter(labels)
    print(f"  Class distribution  :")
    print(f"    Healthy (0)       : {label_counts.get(0, 0)}")
    print(f"    Unhealthy (1)     : {label_counts.get(1, 0)}")

    majority = max(label_counts.values())
    minority = min(label_counts.values())
    ratio = majority / max(minority, 1)
    print(f"  Imbalance ratio     : {ratio:.2f}:1")
    if ratio > 1.5:
        print("  [WARNING] Class imbalance detected! Consider --smote.")
    else:
        print("  [OK] Classes are balanced.")

    if len(label_counts) < 2:
        print("  [ERROR] Only one class found!")
        sys.exit(1)

    lengths = [len(s) for s in sequences]
    print(f"  Sequence lengths    :")
    print(f"    Min               : {min(lengths):,} bp")
    print(f"    Max               : {max(lengths):,} bp")
    print(f"    Mean              : {int(np.mean(lengths)):,} bp")
    print(f"    Std               : {int(np.std(lengths)):,} bp")

    gene_counts = Counter()
    for f in filenames:
        gene = f.split("/")[0]
        gene_counts[gene] += 1
    print(f"  Per-gene counts     :")
    for gene, count in sorted(gene_counts.items()):
        print(f"    {gene:<10}       : {count}")

    n = len(sequences)
    if n < 50:
        print(f"\n  [ADVICE] Very small dataset ({n} samples).")
        print(f"  Recommended minimum: 200+ samples per class.")
        print(f"  Using SMOTE + cross-validation to mitigate.")
    elif n < 200:
        print(f"\n  [ADVICE] Small dataset ({n} samples). CV is essential.")

    print(f"  {'='*60}\n")


# ===========================================================================
# 4. SMOTE OVERSAMPLING
# ===========================================================================

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance training classes."""
    try:
        from imblearn.over_sampling import SMOTE
        print("  Applying SMOTE oversampling...")
        before = Counter(y_train)
        k_neighbors = min(3, min(before.values()) - 1)
        if k_neighbors < 1:
            print("  [!] Too few minority samples for SMOTE, skipping.")
            return X_train, y_train
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        after = Counter(y_res)
        print(f"  Before SMOTE: {dict(before)}")
        print(f"  After SMOTE : {dict(after)}")
        return X_res, y_res
    except ImportError:
        print("  [!] imbalanced-learn not installed. Skipping SMOTE.")
        print("      Install with: pip install imbalanced-learn")
        return X_train, y_train


# ===========================================================================
# 5. MODEL SELECTION
# ===========================================================================

def get_model(model_name: str, class_weights: dict):
    """Return a classifier instance with class-weight balancing."""
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
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
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
    elif model_name == "xgb":
        try:
            from xgboost import XGBClassifier
            scale = class_weights.get(1, 1.0) / max(class_weights.get(0, 1.0), 1e-9)
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
        except ImportError:
            print("  [!] xgboost not installed, falling back to GradientBoosting")
            return get_model("gb", class_weights)
    elif model_name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                class_weight=class_weights,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            print("  [!] lightgbm not installed, falling back to GradientBoosting")
            return get_model("gb", class_weights)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use: rf, svm, lr, gb, xgb, lgbm")


def get_hyperparameter_space(model_name: str) -> dict:
    """Return hyperparameter search space for RandomizedSearchCV."""
    if model_name == "rf":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    elif model_name == "svm":
        return {
            "C": [0.1, 1, 10, 50, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "linear", "poly"],
        }
    elif model_name == "lr":
        return {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        }
    elif model_name == "gb":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
        }
    return {}


# ===========================================================================
# 6. HYPERPARAMETER TUNING
# ===========================================================================

def tune_hyperparameters(
    model, model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    n_iter: int = 30,
) -> object:
    """Run RandomizedSearchCV to find best hyperparameters."""
    param_space = get_hyperparameter_space(model_name)
    if not param_space:
        print(f"  No tuning space for {model_name}, skipping.")
        return model

    print(f"  Running RandomizedSearchCV ({n_iter} iterations)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_space,
        n_iter=min(n_iter, 30),
        cv=cv,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best F1 (CV): {search.best_score_:.4f}")
    print(f"  Best params : {search.best_params_}")
    return search.best_estimator_


# ===========================================================================
# 7. TRAINING
# ===========================================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "rf",
    tune: bool = False,
) -> object:
    """Train the model with class-weight balancing."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    print(f"  Class weights: {class_weight_dict}")

    model = get_model(model_name, class_weight_dict)
    print(f"  Model: {model.__class__.__name__}")

    if tune:
        model = tune_hyperparameters(model, model_name, X_train, y_train)
    else:
        print(f"  Training on {len(y_train)} samples...")
        if model_name == "gb":
            sample_weights = np.array([class_weight_dict[int(y)] for y in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

    print("  Training complete.\n")
    return model


# ===========================================================================
# 8. EVALUATION
# ===========================================================================

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the threshold that maximises F1 score."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.80, 0.02):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "rf",
) -> Dict[str, float]:
    """Evaluate the model with comprehensive per-class metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    print(f"  {'='*60}")
    print(f"  EVALUATION RESULTS — {model.__class__.__name__}")
    print(f"  {'='*60}")

    # --- Class Distribution ---
    true_dist = Counter(y_test)
    pred_dist = Counter(y_pred)
    print(f"\n  --- Class Distribution ---")
    print(f"  y_true : Healthy={true_dist.get(0, 0)}, Unhealthy={true_dist.get(1, 0)}")
    print(f"  y_pred : Healthy={pred_dist.get(0, 0)}, Unhealthy={pred_dist.get(1, 0)}")

    # --- Detect single-class predictions → apply threshold tuning ---
    optimal_thresh = 0.5
    unique_preds = set(y_pred)
    if len(unique_preds) == 1 and y_proba is not None:
        only_class = "Healthy" if 0 in unique_preds else "Unhealthy"
        print(f"\n  [WARNING] Model predicts ONLY '{only_class}'!")
        print(f"  Applying optimal threshold tuning...")
        optimal_thresh = find_optimal_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_thresh).astype(int)
        print(f"  Optimal threshold: {optimal_thresh:.2f}")
        pred_dist = Counter(y_pred)
        print(f"  Adjusted y_pred : Healthy={pred_dist.get(0, 0)}, Unhealthy={pred_dist.get(1, 0)}")

    # --- Per-class and aggregate metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec_h = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_h = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_u = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_u = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  --- Per-Class Metrics ---")
    print(f"  Healthy    : precision={prec_h:.4f}, recall={rec_h:.4f}")
    print(f"  Unhealthy  : precision={prec_u:.4f}, recall={rec_u:.4f}")

    print(f"\n  --- Aggregate Metrics ---")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (macro)      : {f1_macro:.4f}")
    print(f"  F1 (weighted)   : {f1_weighted:.4f}")

    roc_auc = 0.0
    avg_prec = 0.0
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            avg_prec = average_precision_score(y_test, y_proba)
            print(f"  ROC-AUC         : {roc_auc:.4f}")
            print(f"  Avg Precision   : {avg_prec:.4f}")
        except ValueError:
            print(f"  ROC-AUC         : N/A (single class in y_true)")

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
        "accuracy": acc,
        "precision_healthy": prec_h,
        "recall_healthy": rec_h,
        "precision_unhealthy": prec_u,
        "recall_unhealthy": rec_u,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "optimal_threshold": optimal_thresh,
    }


# ===========================================================================
# 9. CROSS-VALIDATION
# ===========================================================================

def run_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, np.ndarray]:
    """Run stratified K-fold cross-validation with multiple metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "roc_auc": "roc_auc",
    }

    print(f"  Running {n_splits}-fold stratified cross-validation...")
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )

    print(f"\n  --- Cross-Validation Results ---")
    for metric_name in scoring:
        key = f"test_{metric_name}"
        scores = results[key]
        print(f"  {metric_name:<20}: {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"[{', '.join(f'{s:.3f}' for s in scores)}]")

    return results


# ===========================================================================
# 10. MULTI-MODEL COMPARISON
# ===========================================================================

def compare_all_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    tune: bool = False,
) -> Dict[str, Dict]:
    """Train and evaluate all available models, return comparison."""
    model_names = ["rf", "svm", "lr", "gb"]

    try:
        import xgboost  # noqa: F401
        model_names.append("xgb")
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401
        model_names.append("lgbm")
    except ImportError:
        pass

    all_results = {}

    for mname in model_names:
        print(f"\n{'='*64}")
        print(f"  MODEL: {mname.upper()}")
        print(f"{'='*64}")

        model = train_model(X_train, y_train, model_name=mname, tune=tune)
        metrics = evaluate_model(model, X_test, y_test, model_name=mname)
        all_results[mname] = {"model": model, "metrics": metrics}

    # --- Summary Table ---
    print(f"\n{'='*64}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*64}")
    print(f"  {'Model':<12} {'Accuracy':>9} {'F1-macro':>9} {'F1-wt':>9} "
          f"{'ROC-AUC':>9} {'Prec(U)':>9} {'Rec(U)':>9} {'Rec(H)':>9}")
    print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

    best_model_name = None
    best_f1 = -1.0

    for mname, data in all_results.items():
        m = data["metrics"]
        print(f"  {mname:<12} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} "
              f"{m['f1_weighted']:>9.4f} {m['roc_auc']:>9.4f} "
              f"{m['precision_unhealthy']:>9.4f} {m['recall_unhealthy']:>9.4f} "
              f"{m['recall_healthy']:>9.4f}")
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_model_name = mname

    print(f"\n  >>> Best model: {best_model_name.upper()} (F1-macro={best_f1:.4f})")
    print(f"{'='*64}\n")

    return all_results


# ===========================================================================
# 11. ENSEMBLE
# ===========================================================================

def build_ensemble(X_train, y_train, X_test, y_test, class_weight_dict):
    """Build a soft-voting ensemble of top models."""
    print(f"\n{'='*64}")
    print(f"  ENSEMBLE MODEL (Soft Voting)")
    print(f"{'='*64}")

    estimators = [
        ("rf", get_model("rf", class_weight_dict)),
        ("lr", get_model("lr", class_weight_dict)),
        ("svm", get_model("svm", class_weight_dict)),
    ]

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    ensemble.fit(X_train, y_train)

    metrics = evaluate_model(ensemble, X_test, y_test, model_name="ensemble")
    return ensemble, metrics


# ===========================================================================
# 12. VISUALIZATION
# ===========================================================================

def save_confusion_matrix_plot(y_test, y_pred, output_path="confusion_matrix.png"):
    """Save confusion matrix as an image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
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
        ax.set_title("Confusion Matrix - Gene Sequence Classification")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix plot saved: {output_path}")
    except ImportError:
        print("  [!] matplotlib/seaborn not installed - skipping plot.")


def save_roc_pr_curves(y_test, y_proba, output_path="roc_pr_curves.png"):
    """Save ROC and Precision-Recall curves side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc_val:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", lw=1)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax2.plot(recall, precision, color="green", lw=2, label=f"PR (AP={ap:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  ROC/PR curves saved: {output_path}")
    except (ImportError, ValueError) as e:
        print(f"  [!] Could not generate curves: {e}")


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
        help="K-mer size (default: 4, 256 features)",
    )
    parser.add_argument(
        "--model", default="rf", choices=["rf", "svm", "lr", "gb", "xgb", "lgbm", "all"],
        help="Model to use (or 'all' to compare all models)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
        help="Held-out test fraction (default: 0.3); five-fold CV uses training data only",
    )
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to balance classes",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Hyperparameter tuning via RandomizedSearchCV",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Build soft-voting ensemble of RF + LR + SVM",
    )
    parser.add_argument(
        "--feature-select", type=int, default=0,
        help="Select top N features by mutual information (0=all)",
    )
    parser.add_argument("--compare-features", action="store_true",
                        help="Compare four feature sets on identical training-only validation folds")
    parser.add_argument("--audit-only", action="store_true",
                        help="Print an exact-sequence duplicate/conflict audit without training")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("  GENE SEQUENCE CLASSIFICATION PIPELINE")
    print("  (Healthy vs Unhealthy - APP, PSEN1, PSEN2)")
    print("=" * 64)
    print(f"\n  Dataset path   : {args.dataset}")
    print(f"  K-mer size     : {args.kmer}")
    print(f"  Model          : {args.model}")
    print(f"  Test size      : {args.test_size}")
    print(f"  SMOTE          : {'ON' if args.smote else 'OFF'}")
    print(f"  Tuning         : {'ON' if args.tune else 'OFF'}")
    print(f"  Ensemble       : {'ON' if args.ensemble else 'OFF'}")
    print(f"  Feature select : {args.feature_select if args.feature_select > 0 else 'ALL'}")

    # ---- Step 1: Load Data ----
    print(f"\n{'-'*64}")
    print("  STEP 1: Loading FASTA sequences...")
    print(f"{'-'*64}")
    sequences, labels, filenames, gene_names = load_data(args.dataset)

    if not sequences:
        print("  [ERROR] No sequences loaded! Check dataset path.")
        print(f"  Expected: {args.dataset}/APP/output_safe_healthy/*.fasta")
        sys.exit(1)

    # ---- Step 2: Data Checks ----
    data_checks(sequences, labels, filenames)

    from validation_pipeline import audit_dataset, run_experiment
    if args.audit_only:
        import json
        print(json.dumps(audit_dataset(sequences, labels, gene_names, filenames), indent=2))
        return
    try:
        result = run_experiment(
            sequences, labels, gene_names, k=args.kmer, model_name=args.model,
            test_size=args.test_size, smote=args.smote, tune=args.tune,
            feature_select=args.feature_select, ensemble=args.ensemble,
            compare_features=args.compare_features, filenames=filenames,
        )
    except (ValueError, ImportError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)
    metrics = result["metrics"]
    print(f"\n  Split: {result['split']['train']} training / {result['split']['test']} held-out test")
    print(f"  Gene/class counts: {result['split']['genes']}")
    print(f"  Selected by mean validation F1: {result['best_model']} / {result['best_features']}")
    print(f"  Dataset audit: {result['audit']}")
    for row in result["comparison"]:
        print(f"\n  Five-fold validation: {row['name']} / {row['feature_set']} ({row['features']} features)")
        for fold in row["folds"]:
            print(f"    Fold {fold['fold']}: train={fold['train']}, validation={fold['validation']}, "
                  f"accuracy={fold['accuracy']:.4f}, F1={fold['f1_macro']:.4f}, AUC={fold['roc_auc']:.4f}")
        for metric, deviation in row["std"].items():
            print(f"    {metric}: {row[metric]:.4f} +/- {deviation:.4f}")
    print(f"  Per-gene held-out results: {result['per_gene']}")
    print("  Research evaluation only; related/generated sequences may inflate generalization estimates.")
    save_confusion_matrix_plot(result["test_labels"], result["predictions"])
    save_roc_pr_curves(result["test_labels"], result["probabilities"])
    for feature in result["top_features"]:
        print(f"  {feature['name']}: {feature['importance']:.5f}")

    # ---- Final Summary ----
    print("=" * 64)
    print("  PIPELINE COMPLETE")
    print("=" * 64)
    print(f"\n  Held-out Test Results ({result['best_model']}):")
    print(f"    Accuracy          = {metrics['accuracy']:.4f}")
    print(f"    F1 (macro)        = {metrics['f1_macro']:.4f}")
    print(f"    F1 (weighted)     = {metrics['f1_weighted']:.4f}")
    print(f"    ROC-AUC           = {metrics['roc_auc']:.4f}")
    print(f"    Healthy recall    = {metrics['recall_healthy']:.4f}")
    print(f"    Unhealthy recall  = {metrics['recall_unhealthy']:.4f}")

    # ---- Recommendations ----
    print(f"\n  {'-'*60}")
    print(f"  RECOMMENDATIONS FOR MEDICAL CLASSIFICATION")
    print(f"  {'-'*60}")
    print(f"  1. Primary metric: F1-macro (balances both classes)")
    print(f"  2. For medical safety: maximise Recall(Unhealthy)")
    print(f"     = minimise false negatives (missing pathogenic variants)")
    print(f"  3. Dataset has {len(sequences)} samples. Min recommended: 200+ per class")
    print(f"  4. Use --smote for class imbalance")
    print(f"  5. Use --tune for hyperparameter optimisation")
    print(f"  6. Use --model all to compare algorithms")
    print(f"  7. Use --ensemble for soft-voting ensemble")
    print(f"  8. Data augmentation ideas: reverse complement,")
    print(f"     subsequence sampling, simulated mutations")
    print()


if __name__ == "__main__":
    main()
