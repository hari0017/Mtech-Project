"""
dnabert_model.py — DNABERT model loading, classification head, and inference.

Wraps HuggingFace Transformers to:
  1. Load a pretrained DNABERT (or any compatible BERT) checkpoint.
  2. Attach a binary classification head (pathogenic vs. benign).
  3. Run inference on encoded DNA sequences.
  4. Provide basic evaluation metrics (accuracy, F1).

Checkpoint loading errors are fatal; random models are not substituted for
pretrained models. Newly initialized classification heads require fine-tuning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from collections import Counter

logger = logging.getLogger("pangenome_pipeline")


# ---------------------------------------------------------------------------
# Try to import transformers; give a clear message if missing
# ---------------------------------------------------------------------------

try:
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer,
    )

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning(
        "HuggingFace transformers not installed — "
        "model functionality will be limited. "
        "Install with: pip install transformers"
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Container for a single sequence prediction.

    Attributes:
        sequence_index: Index of the input sequence in the batch.
        logits: Raw model outputs (one per class).
        probabilities: Softmax probabilities (one per class).
        predicted_class: Argmax class index.
        predicted_label: Human-readable label string.
    """
    sequence_index: int
    logits: List[float]
    probabilities: List[float]
    predicted_class: int
    predicted_label: str


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[int, str] = {
    0: "BENIGN",
    1: "PATHOGENIC",
}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class DNABertClassifier:
    """Wrapper around a BERT-based model for variant pathogenicity prediction.

    Args:
        model_name: HuggingFace model identifier or local path.
            Common choices:
            * ``zhihan1996/DNA_bert_3`` (DNABERT with k=3)
            * ``zhihan1996/DNA_bert_6`` (DNABERT with k=6)
        num_labels: Number of output classes (default 2).
        vocab_size: Checkpoint tokenizer vocabulary size, validated against
            the model's embedding vocabulary.
        max_seq_length: Maximum sequence length the model should accept.
        device: ``"cpu"`` or ``"cuda"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        model_name: str = "zhihan1996/DNA_bert_3",
        num_labels: int = 2,
        vocab_size: int = 4101,
        max_seq_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers is required.  pip install transformers"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = self._load_model(model_name, num_labels, vocab_size)
        if max_seq_length < 3 or max_seq_length > self.model.config.max_position_embeddings:
            raise ValueError("Sequence length must be at least 3 and within the checkpoint's position limit.")
        if vocab_size != self.model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match the model checkpoint.")
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "DNABertClassifier ready on %s  (params: %s)",
            self.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    # ---- model loading ----

    def _load_model(
        self,
        model_name: str,
        num_labels: int,
        vocab_size: int,
    ) -> BertForSequenceClassification:
        """Load a checkpoint without silently substituting a random model."""
        try:
            logger.info("Attempting to load pretrained model '%s' ...", model_name)
            model, loading_info = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                output_loading_info=True,
            )
            self.has_trained_head = not any(
                "classifier" in key for key in loading_info.get("missing_keys", [])
            )
            if not self.has_trained_head:
                logger.warning("Classification head is newly initialized; fine-tuning is required.")
            logger.info("Successfully loaded pretrained model")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load checkpoint '{model_name}'. Verify the path, network access "
                "and model compatibility. No random-model fallback was used."
            ) from exc

    # ---- inference ----

    @torch.no_grad()
    def predict(
        self,
        token_ids_batch: List[List[int]],
        attention_mask_batch: List[List[int]],
    ) -> List[PredictionResult]:
        """Run inference on a batch of encoded sequences.

        Args:
            token_ids_batch: List of integer token-ID sequences.
            attention_mask_batch: Corresponding attention masks.

        Returns:
            A :class:`PredictionResult` for every input sequence.
        """
        if not self.has_trained_head:
            raise ValueError("Classification head is untrained. Fine-tune with --train and verified labels first.")
        input_ids = torch.tensor(token_ids_batch, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            attention_mask_batch, dtype=torch.long, device=self.device
        )

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()  # (batch, num_labels)

        results: List[PredictionResult] = []
        for i, logit_row in enumerate(logits):
            probs = _softmax(logit_row)
            pred_cls = int(np.argmax(probs))
            results.append(
                PredictionResult(
                    sequence_index=i,
                    logits=logit_row.tolist(),
                    probabilities=probs.tolist(),
                    predicted_class=pred_cls,
                    predicted_label=LABEL_MAP.get(pred_cls, f"CLASS_{pred_cls}"),
                )
            )

        return results

    def predict_single(
        self,
        token_ids: List[int],
        attention_mask: List[int],
    ) -> PredictionResult:
        """Convenience: predict on a single sequence."""
        return self.predict([token_ids], [attention_mask])[0]

    # ---- training / fine-tuning ----

    def train_model(
        self,
        token_ids_batch: List[List[int]],
        attention_mask_batch: List[List[int]],
        labels: List[int],
        epochs: int = 10,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        val_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """Fine-tune the model on labelled data with class-weight balancing.

        Args:
            token_ids_batch: List of integer token-ID sequences.
            attention_mask_batch: Corresponding attention masks.
            labels: Integer class labels (0=BENIGN, 1=PATHOGENIC).
            epochs: Number of training epochs.
            learning_rate: Learning rate for AdamW optimizer.
            batch_size: Mini-batch size.
            val_split: Fraction of data used for validation (0 to disable).

        Returns:
            Dictionary with training history (train_loss, val_loss, val_acc, val_f1).
        """
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("Epochs, batch size and learning rate must be positive.")
        if not labels or len(labels) != len(token_ids_batch) or len(labels) != len(attention_mask_batch):
            raise ValueError("Training requires nonempty, aligned inputs, masks and labels.")
        input_ids = torch.tensor(token_ids_batch, dtype=torch.long)
        attn_mask = torch.tensor(attention_mask_batch, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        n = len(labels)
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
        }

        # Compute class weights to handle imbalance
        class_counts = Counter(labels)
        num_classes = self.num_labels
        class_weights = torch.ones(num_classes, device=self.device)
        for cls_id, count in class_counts.items():
            class_weights[cls_id] = n / (num_classes * count)
        logger.info(
            "Class weights: %s (counts: %s)",
            {LABEL_MAP[i]: f"{class_weights[i]:.3f}" for i in range(num_classes)},
            {LABEL_MAP.get(k, k): v for k, v in class_counts.items()},
        )
        self._class_weights = class_weights

        # Train/val split
        val_size = int(n * val_split) if val_split > 0 else 0
        if val_size > 0 and val_size < n:
            # Shuffle indices deterministically
            perm = torch.randperm(n)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]

            train_dataset = TensorDataset(
                input_ids[train_idx], attn_mask[train_idx], label_tensor[train_idx]
            )
            val_dataset = TensorDataset(
                input_ids[val_idx], attn_mask[val_idx], label_tensor[val_idx]
            )
        else:
            train_dataset = TensorDataset(input_ids, attn_mask, label_tensor)
            val_dataset = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Weighted loss function for class imbalance
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        logger.info(
            "Starting training: %d samples, %d epochs, lr=%.1e, batch_size=%d",
            len(train_dataset), epochs, learning_rate, batch_size,
        )

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            num_batches = 0

            for batch_ids, batch_mask, batch_labels in train_loader:
                batch_ids = batch_ids.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                )
                # Use weighted loss instead of default unweighted loss
                loss = loss_fn(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_dataset is not None:
                val_metrics = self._validate(val_dataset, batch_size)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_f1"].append(val_metrics["f1"])
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, "
                    "val_acc: %.4f, val_f1: %.4f",
                    epoch, epochs, avg_train_loss,
                    val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"],
                )
            else:
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f",
                    epoch, epochs, avg_train_loss,
                )

        self.model.eval()
        self.has_trained_head = True
        logger.info("Training complete")
        return history

    @torch.no_grad()
    def _validate(
        self, val_dataset: TensorDataset, batch_size: int
    ) -> Dict[str, float]:
        """Run validation on a held-out set."""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        all_preds, all_labels = [], []
        total_loss = 0.0
        num_batches = 0

        for batch_ids, batch_mask, batch_labels in val_loader:
            batch_ids = batch_ids.to(self.device)
            batch_mask = batch_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)

            outputs = self.model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                labels=batch_labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().tolist())

        self.model.train()
        avg_loss = total_loss / max(num_batches, 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        return {"loss": avg_loss, "accuracy": acc, "f1": f1}

    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model and config to disk."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info("Model saved to '%s'", output_dir)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(
    true_labels: List[int],
    predicted_labels: List[int],
    print_details: bool = True,
) -> Dict[str, float]:
    """Compute classification metrics with full diagnostics.

    Args:
        true_labels: Ground-truth class indices.
        predicted_labels: Model-predicted class indices.
        print_details: Whether to print confusion matrix and report.

    Returns:
        Dictionary with accuracy, f1, precision, and recall.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Label count mismatch: {len(true_labels)} true vs "
            f"{len(predicted_labels)} predicted"
        )
    if not true_labels:
        logger.warning("Empty label lists - returning zeroed metrics")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="binary", zero_division=0)
    prec = precision_score(true_labels, predicted_labels, average="binary", zero_division=0)
    rec = recall_score(true_labels, predicted_labels, average="binary", zero_division=0)

    if print_details:
        # Class distribution
        true_dist = Counter(true_labels)
        pred_dist = Counter(predicted_labels)
        print(f"\n  --- Class Distribution ---")
        print(f"  y_true : BENIGN={true_dist.get(0, 0)}, PATHOGENIC={true_dist.get(1, 0)}")
        print(f"  y_pred : BENIGN={pred_dist.get(0, 0)}, PATHOGENIC={pred_dist.get(1, 0)}")

        # Check single-class prediction
        unique_preds = set(predicted_labels)
        if len(unique_preds) == 1:
            only_class = LABEL_MAP.get(list(unique_preds)[0], "UNKNOWN")
            print(f"\n  [WARNING] Model predicts ONLY '{only_class}' for all samples!")
            print(f"            This indicates the model has not learned to distinguish classes.")

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
        print(f"\n  --- Confusion Matrix ---")
        print(f"                    Predicted")
        print(f"                    BENIGN  PATHOGENIC")
        print(f"  Actual BENIGN     {cm[0][0]:<7} {cm[0][1]}")
        print(f"  Actual PATHOGENIC {cm[1][0]:<7} {cm[1][1]}")

        # Classification report
        target_names = ["BENIGN", "PATHOGENIC"]
        report = classification_report(
            true_labels, predicted_labels,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
        print(f"\n  --- Classification Report ---")
        for line in report.split("\n"):
            print(f"  {line}")

    logger.info(
        "Evaluation - accuracy: %.4f, F1: %.4f, precision: %.4f, recall: %.4f",
        acc, f1, prec, rec,
    )
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def predict_with_threshold(
    results: List["PredictionResult"],
    threshold: float = 0.5,
) -> List[int]:
    """Apply a custom threshold to probability predictions.

    By default, argmax uses 0.5 as threshold. Lowering it makes the model
    more sensitive to the minority class (PATHOGENIC).

    Args:
        results: List of PredictionResult from model inference.
        threshold: Probability threshold for class 1 (PATHOGENIC).
            Values < 0.5 make the model predict more PATHOGENIC.

    Returns:
        List of predicted class indices using the custom threshold.
    """
    adjusted_preds = []
    for r in results:
        prob_pathogenic = r.probabilities[1]
        adjusted_preds.append(1 if prob_pathogenic >= threshold else 0)
    return adjusted_preds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
