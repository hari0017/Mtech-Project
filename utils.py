"""
utils.py — Utility functions for the pangenome-to-DNABERT pipeline.

Provides logging setup, configuration management, file I/O helpers,
and validation routines used throughout the project.
"""

import logging
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Central configuration for every stage of the pipeline.

    Attributes:
        gfa_path: Path to the input GFA / rGFA file.
        kmer_size: Length of k-mers for tokenization (3 or 6 typical).
        max_seq_length: Maximum number of tokens per encoded sequence
            (including special tokens).
        max_paths: Upper bound on how many paths to extract from the DAG.
        random_walk_paths: Number of additional random-walk paths to sample.
        random_seed: Seed for reproducible random walks.
        model_name: HuggingFace model identifier for DNABERT or compatible.
        num_labels: Number of output classes (2 for pathogenic/benign).
        cls_token: Classification special token.
        sep_token: Separator special token.
        pad_token: Padding special token.
    """

    gfa_path: str = "sample.gfa"
    kmer_size: int = 3
    max_seq_length: int = 128
    max_paths: int = 10
    random_walk_paths: int = 3
    random_seed: int = 42
    model_name: str = "zhihan1996/DNA_bert_3"
    num_labels: int = 2
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    # Label / annotation file (CSV, TSV, or VCF)
    labels_file: str = ""
    # Training mode
    train: bool = False
    epochs: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 8
    val_split: float = 0.2
    # Output directory for saved models
    output_dir: str = "trained_model"
    test_split: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "pangenome_pipeline",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name (appears in log records).
        level: Logging level (DEBUG, INFO, WARNING, …).
        log_file: If provided, log messages are also written to this file.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dna_sequence(sequence: str, allow_n: bool = True) -> bool:
    """Return True if *sequence* contains only valid DNA characters.

    Args:
        sequence: The DNA string to validate.
        allow_n: Whether to accept 'N' (unknown base) as valid.

    Returns:
        ``True`` when every character is in {A, C, G, T} (and optionally N).
    """
    valid_bases = set("ACGTacgt")
    if allow_n:
        valid_bases.update("Nn")
    return all(ch in valid_bases for ch in sequence)


def validate_file_exists(path: str) -> None:
    """Raise FileNotFoundError if *path* does not exist."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")


# ---------------------------------------------------------------------------
# Printing / display helpers
# ---------------------------------------------------------------------------

def print_section(title: str, width: int = 72) -> None:
    """Print a visible section header to stdout."""
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
