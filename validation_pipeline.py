from collections import Counter
from hashlib import sha256
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler, RandomizedSearchCV, StratifiedKFold, cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from gene_classifier import (
    build_kmer_vocab, get_hyperparameter_space, get_model, preprocess,
)


class GeneFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, k=4, feature_set="all"):
        self.k = k
        self.feature_set = feature_set

    def fit(self, records, labels):
        self.references_ = {}
        for (sequence, gene), label in zip(records, labels):
            if label == 0 and gene not in self.references_:
                self.references_[gene] = sequence
        missing = set(records[:, 1]) - set(self.references_)
        if missing:
            raise ValueError(f"Training fold needs a healthy reference for {sorted(missing)}")
        return self

    def transform(self, records):
        if self.feature_set in ("kmer", "composition"):
            features = preprocess(records[:, 0].tolist(), k=self.k)
            return features[:, :4 ** self.k] if self.feature_set == "kmer" else features
        features = preprocess(
            records[:, 0].tolist(), k=self.k, gene_names=records[:, 1].tolist(),
            references=self.references_,
        )
        return features[:, :-3] if self.feature_set == "mutation" else features


class FoldClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model_name="rf", smote=False, parameters=None):
        self.model_name = model_name
        self.smote = smote
        self.parameters = parameters

    def fit(self, features, labels):
        if self.smote:
            from imblearn.over_sampling import SMOTE
            minority = min(Counter(labels).values())
            if minority < 2:
                raise ValueError("SMOTE needs at least two training samples per class.")
            features, labels = SMOTE(
                random_state=42, k_neighbors=min(3, minority - 1),
            ).fit_resample(features, labels)
        self.classes_ = np.unique(labels)
        weights = compute_class_weight("balanced", classes=self.classes_, y=labels)
        class_weights = dict(zip(self.classes_, weights))
        if self.model_name == "ensemble":
            from sklearn.ensemble import VotingClassifier
            self.estimator_ = VotingClassifier(
                [(name, get_model(name, class_weights)) for name in ("rf", "lr", "svm")],
                voting="soft", n_jobs=1,
            )
        else:
            self.estimator_ = get_model(self.model_name, class_weights)
        if self.parameters:
            self.estimator_.set_params(**self.parameters)
        if self.model_name == "gb":
            self.estimator_.fit(features, labels, sample_weight=[class_weights[label] for label in labels])
        else:
            self.estimator_.fit(features, labels)
        return self

    def predict(self, features):
        return self.estimator_.predict(features)

    def predict_proba(self, features):
        return self.estimator_.predict_proba(features)


def feature_names(k, feature_set="all"):
    mutation_names = [
        "Mutation count", "Mean mutation position", "Mutation position std",
        "First mutation position", "Last mutation position", "Transitions",
        "Transversions", "Local GC", "Mutation region", "Context entropy", "Length difference",
    ]
    names = (
        list(build_kmer_vocab(k)) + ["A%", "C%", "G%", "T%", "GC%", "AT/GC"]
        + mutation_names + [f"Context {word}" for word in build_kmer_vocab(min(k, 4))]
        + ["Gene APP", "Gene PSEN1", "Gene PSEN2"]
    )
    if feature_set == "kmer":
        return names[:4 ** k]
    if feature_set == "composition":
        return names[:4 ** k + 6]
    if feature_set == "mutation":
        return names[:-3]
    return names


def audit_dataset(sequences, labels, genes, filenames=None, train_indices=None, test_indices=None):
    if not (len(sequences) == len(labels) == len(genes)):
        raise ValueError("Audit inputs must have equal lengths.")
    filenames = filenames or [str(index) for index in range(len(sequences))]
    if len(filenames) != len(sequences):
        raise ValueError("Audit requires one filename per sequence.")
    groups = {}
    for index, sequence in enumerate(sequences):
        digest = sha256(sequence.upper().encode("ascii")).hexdigest()
        groups.setdefault(digest, []).append(index)
    train_set = set([] if train_indices is None else train_indices)
    test_set = set([] if test_indices is None else test_indices)
    duplicates = []
    for digest, indices in groups.items():
        if len(indices) < 2:
            continue
        duplicates.append({
            "sha256": digest, "count": len(indices),
            "conflicting_labels": len({labels[index] for index in indices}) > 1,
            "crosses_holdout": bool(train_set.intersection(indices) and test_set.intersection(indices)),
            "members": [{"file": filenames[index], "gene": genes[index], "label": int(labels[index])}
                        for index in indices],
        })
    return {
        "total": len(sequences), "unique_sequences": len(groups),
        "duplicate_groups": len(duplicates),
        "duplicate_extra_copies": sum(group["count"] - 1 for group in duplicates),
        "conflicting_groups": sum(group["conflicting_labels"] for group in duplicates),
        "holdout_overlap_groups": sum(group["crosses_holdout"] for group in duplicates),
        "split_checked": train_indices is not None and test_indices is not None,
        "groups": duplicates,
        "scope": "Exact uppercase sequence matches after the training loader's cleaning. No near-duplicate, provenance or clinical-label validation. No files were modified.",
    }


def score_predictions(labels, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "avg_precision": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "precision_healthy": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "precision_unhealthy": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_healthy": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "recall_unhealthy": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
    }


def run_experiment(sequences, labels, genes, k=4, model_name="rf", test_size=0.3,
                   smote=False, tune=False, feature_select=0, ensemble=False,
                   compare_features=False, filenames=None):
    if k not in (3, 4, 5, 6) or not 0.1 <= test_size <= 0.4:
        raise ValueError("K-mer size must be 3-6 and test fraction must be 0.1-0.4.")
    if feature_select < 0:
        raise ValueError("Feature selection count cannot be negative.")
    if not sequences or not (len(sequences) == len(labels) == len(genes)):
        raise ValueError("Sequences, labels and genes must be nonempty and have equal lengths.")
    if set(labels) != {0, 1}:
        raise ValueError("Both healthy (0) and unhealthy (1) labels are required.")
    records = np.array(list(zip(sequences, genes)), dtype=object)
    targets = np.asarray(labels)
    strata = np.array([f"{gene}:{label}" for gene, label in zip(genes, labels)])
    train_indices, test_indices = train_test_split(
        np.arange(len(records)), test_size=test_size, stratify=strata, random_state=42,
    )
    if min(Counter(strata[train_indices]).values()) < 5:
        raise ValueError("Five-fold validation requires at least five training samples per gene/class.")
    train_records, train_labels = records[train_indices], targets[train_indices]
    audit = audit_dataset(sequences, labels, genes, filenames, train_indices, test_indices)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        train_records, strata[train_indices],
    ))
    scoring = {
        "accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc",
        "recall_unhealthy": make_scorer(recall_score, pos_label=1, zero_division=0),
        "recall_healthy": make_scorer(recall_score, pos_label=0, zero_division=0),
    }
    names = ["rf", "svm", "lr", "gb"] if model_name == "all" else [model_name]
    if model_name == "all":
        from importlib.util import find_spec
        names += [name for name, module in (("xgb", "xgboost"), ("lgbm", "lightgbm")) if find_spec(module)]
    if ensemble:
        names.append("ensemble")
    comparison, candidates, tuning = [], {}, {}
    feature_sets = ("kmer", "composition", "mutation", "all") if compare_features else ("all",)
    for name, feature_set in product(names, feature_sets):
        steps = [("features", GeneFeatures(k, feature_set))]
        if feature_select:
            steps.append(("select", SelectKBest(mutual_info_classif, k=min(feature_select, len(feature_names(k, feature_set))))))
        steps += [("scale", StandardScaler()), ("model", FoldClassifier(name, smote))]
        pipeline = Pipeline(steps)
        space = get_hyperparameter_space(name) if tune else {}
        if space:
            parameter_sets = list(ParameterSampler(space, n_iter=10, random_state=42))
            search = RandomizedSearchCV(
                pipeline, {"model__parameters": parameter_sets}, n_iter=len(parameter_sets),
                scoring=scoring, refit="f1_macro", cv=folds, n_jobs=1,
                random_state=42, error_score="raise",
            )
            search.fit(train_records, train_labels)
            best_index = search.best_index_
            scores = {f"test_{metric}": np.array([
                search.cv_results_[f"split{fold}_test_{metric}"][best_index] for fold in range(5)
            ]) for metric in scoring}
            pipeline = search.best_estimator_
            tuning[f"{name}:{feature_set}"] = search.best_params_
        else:
            scores = cross_validate(pipeline, train_records, train_labels, cv=folds,
                                    scoring=scoring, n_jobs=1, error_score="raise")
        summary = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring}
        summary["std"] = {metric: float(np.std(scores[f"test_{metric}"])) for metric in scoring}
        summary["name"] = name.upper()
        summary["feature_set"] = feature_set
        summary["features"] = min(feature_select or len(feature_names(k, feature_set)), len(feature_names(k, feature_set)))
        summary["folds"] = [dict(
            fold=fold + 1, train=len(fold_train), validation=len(fold_valid),
            **{metric: float(scores[f"test_{metric}"][fold]) for metric in scoring},
        ) for fold, (fold_train, fold_valid) in enumerate(folds)]
        comparison.append(summary)
        candidates[(name, feature_set)] = pipeline
        print(f"  {name.upper()} / {feature_set} CV F1-macro: {summary['f1_macro']:.4f} +/- {summary['std']['f1_macro']:.4f}")
    winner = max(comparison, key=lambda row: row["f1_macro"])
    best_name = winner["name"].lower()
    pipeline = candidates[(best_name, winner["feature_set"])]
    pipeline.fit(train_records, train_labels)
    test_records, test_labels = records[test_indices], targets[test_indices]
    predictions = pipeline.predict(test_records)
    probabilities = pipeline.predict_proba(test_records)[:, 1]
    metrics = score_predictions(test_labels, predictions, probabilities)
    split = {"train": len(train_indices), "test": len(test_indices), "test_fraction": test_size,
             "seed": 42, "folds": 5, "stratification": "gene + class", "genes": {}}
    for gene in sorted(set(genes)):
        split["genes"][gene] = {
            portion: {"healthy": int(sum((records[indices, 1] == gene) & (targets[indices] == 0))),
                      "unhealthy": int(sum((records[indices, 1] == gene) & (targets[indices] == 1)))}
            for portion, indices in (("train", train_indices), ("test", test_indices))
        }
    per_gene = {}
    for gene in sorted(set(genes)):
        mask = test_records[:, 1] == gene
        per_gene[gene] = dict(samples=int(sum(mask)), **score_predictions(
            test_labels[mask], predictions[mask], probabilities[mask],
        ))
    estimator = pipeline.named_steps["model"].estimator_
    names = feature_names(k, winner["feature_set"])
    if feature_select:
        names = np.asarray(names)[pipeline.named_steps["select"].get_support()].tolist()
    top_features = []
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        top_features = [{"name": names[index], "importance": float(importance[index])}
                        for index in np.argsort(importance)[-10:][::-1]]
    return {
        "pipeline": pipeline, "metrics": metrics, "comparison": comparison,
        "best_features": winner["feature_set"], "audit": audit,
        "best_model": winner["name"], "cv": winner, "split": split,
        "per_gene": per_gene, "top_features": top_features,
        "feature_shape": [len(records), len(names)], "tuning": tuning,
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist(),
        "test_labels": test_labels, "predictions": predictions, "probabilities": probabilities,
        "train_indices": train_indices, "test_indices": test_indices,
        "fold_indices": [(train_indices[train], train_indices[valid]) for train, valid in folds],
    }