"""
annotation_loader.py — Load variant labels from CSV or VCF annotation files.

Supports two formats:
  1. CSV/TSV with columns: sequence_id, label (BENIGN/PATHOGENIC or 0/1)
  2. VCF files with ClinVar CLNSIG annotations in the INFO field.

The loader produces a mapping from sequence/path index to integer label
(0 = BENIGN, 1 = PATHOGENIC).
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pangenome_pipeline")

# Accepted label synonyms mapped to canonical integers
_LABEL_ALIASES: Dict[str, int] = {
    # Benign variants
    "benign": 0,
    "likely_benign": 0,
    "likely benign": 0,
    "0": 0,
    # Pathogenic variants
    "pathogenic": 1,
    "likely_pathogenic": 1,
    "likely pathogenic": 1,
    "pathogenic/likely_pathogenic": 1,
    "1": 1,
}


def _normalise_label(raw: str) -> Optional[int]:
    """Convert a raw label string to an integer class, or None if unknown."""
    key = raw.strip().lower().replace("/", "/").replace("_", "_")
    return _LABEL_ALIASES.get(key)


# ---------------------------------------------------------------------------
# CSV / TSV loader
# ---------------------------------------------------------------------------

def load_labels_csv(filepath: str) -> Dict[int, int]:
    """Load labels from a CSV or TSV file.

    Expected columns (header required):

    +--------------+-----------------------------+
    | Column       | Description                 |
    +==============+=============================+
    | sequence_id  | 0-based integer index of    |
    |              | the sequence / path          |
    +--------------+-----------------------------+
    | label        | BENIGN, PATHOGENIC, 0, or 1 |
    +--------------+-----------------------------+

    Args:
        filepath: Path to the CSV / TSV file.

    Returns:
        Mapping from sequence index (int) to label (0 or 1).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")

    labels: Dict[int, int] = {}

    with open(filepath, "r", encoding="utf-8") as fh:
        # Auto-detect delimiter
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample else ","

        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV file: {filepath}")

        # Normalise header names
        field_map = {f.strip().lower(): f for f in reader.fieldnames}

        id_col = field_map.get("sequence_id") or field_map.get("path_id") or field_map.get("id")
        label_col = field_map.get("label") or field_map.get("class") or field_map.get("clnsig")

        if id_col is None or label_col is None:
            raise ValueError(
                f"CSV must contain 'sequence_id' (or 'path_id'/'id') and "
                f"'label' (or 'class'/'clnsig') columns. "
                f"Found: {list(reader.fieldnames)}"
            )

        skipped = 0
        for row_num, row in enumerate(reader, start=2):
            raw_id = row[id_col].strip()
            raw_label = row[label_col].strip()

            try:
                seq_idx = int(raw_id)
            except ValueError:
                logger.warning("Row %d: non-integer sequence_id '%s', skipping", row_num, raw_id)
                skipped += 1
                continue

            label = _normalise_label(raw_label)
            if label is None:
                logger.warning(
                    "Row %d: unrecognised label '%s' for sequence %d, skipping",
                    row_num, raw_label, seq_idx,
                )
                skipped += 1
                continue

            labels[seq_idx] = label

    logger.info(
        "Loaded %d labels from '%s' (%d rows skipped)",
        len(labels), filepath, skipped,
    )
    return labels


# ---------------------------------------------------------------------------
# VCF loader (ClinVar-style)
# ---------------------------------------------------------------------------

def load_labels_vcf(filepath: str) -> Dict[int, int]:
    """Load labels from a VCF file with ClinVar CLNSIG annotations.

    The loader reads the INFO field for ``CLNSIG=<value>`` and maps each
    variant to a label.  Variants are indexed by their order in the file
    (0-based).

    Args:
        filepath: Path to the VCF file.

    Returns:
        Mapping from variant index (int) to label (0 or 1).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"VCF file not found: {filepath}")

    labels: Dict[int, int] = {}
    variant_idx = 0
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                skipped += 1
                continue

            info_field = fields[7]
            clnsig = None
            for entry in info_field.split(";"):
                if entry.startswith("CLNSIG="):
                    clnsig = entry.split("=", 1)[1]
                    break

            if clnsig is not None:
                label = _normalise_label(clnsig)
                if label is not None:
                    labels[variant_idx] = label
                else:
                    logger.debug(
                        "Variant %d: unrecognised CLNSIG '%s', skipping",
                        variant_idx, clnsig,
                    )
                    skipped += 1
            else:
                skipped += 1

            variant_idx += 1

    logger.info(
        "Loaded %d labels from VCF '%s' (%d variants skipped)",
        len(labels), filepath, skipped,
    )
    return labels


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_labels(filepath: str) -> Dict[int, int]:
    """Auto-detect file format and load labels.

    Supported extensions: ``.csv``, ``.tsv``, ``.vcf``, ``.vcf.gz`` (plain).

    Args:
        filepath: Path to the label / annotation file.

    Returns:
        Mapping from sequence index to label integer.
    """
    ext = filepath.lower()
    if ext.endswith(".vcf") or ext.endswith(".vcf.gz"):
        return load_labels_vcf(filepath)
    else:
        return load_labels_csv(filepath)


def get_labels_for_sequences(
    label_map: Dict[int, int],
    num_sequences: int,
) -> Tuple[List[int], List[int]]:
    """Build aligned label and index lists for the given number of sequences.

    Only sequences that have an entry in *label_map* are included.

    Args:
        label_map: Mapping from sequence index to label.
        num_sequences: Total number of sequences in the pipeline.

    Returns:
        A tuple ``(indices, labels)`` where *indices* are the 0-based
        sequence positions that have labels, and *labels* are the
        corresponding integer labels.
    """
    indices: List[int] = []
    labels: List[int] = []
    for i in range(num_sequences):
        if i in label_map:
            indices.append(i)
            labels.append(label_map[i])

    missing = num_sequences - len(indices)
    if missing > 0:
        logger.warning(
            "%d of %d sequences have no label — they will be skipped in "
            "training/evaluation",
            missing, num_sequences,
        )

    return indices, labels
