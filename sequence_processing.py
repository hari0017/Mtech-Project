"""
sequence_processing.py — Linearize paths into DNA sequences and encode them
for transformer input.

Responsibilities:
  1. Concatenate node DNA sequences along a path.
  2. Add special tokens ([CLS], [SEP], [PAD]).
  3. Pad or truncate token lists to a fixed length.
  4. Generate positional index arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

logger = logging.getLogger("pangenome_pipeline")


# ---------------------------------------------------------------------------
# Sequence linearization
# ---------------------------------------------------------------------------

def linearize_path(dag: nx.DiGraph, path: List[str]) -> str:
    """Concatenate the DNA sequences of every node along *path*.

    Args:
        dag: DAG whose nodes carry a ``sequence`` attribute.
        path: Ordered list of node names forming the path.

    Returns:
        A single DNA string.

    Raises:
        KeyError: If a node in *path* is not present in *dag*.
    """
    fragments: List[str] = []
    for node in path:
        if node not in dag.nodes:
            raise KeyError(f"Node '{node}' not found in graph")
        seq = dag.nodes[node].get("sequence", "")
        if seq and seq != "*":
            fragments.append(seq)
    concatenated = "".join(fragments)
    logger.debug(
        "Linearized path (%d nodes) → %d bp", len(path), len(concatenated)
    )
    return concatenated


def linearize_paths(
    dag: nx.DiGraph,
    paths: List[List[str]],
) -> List[str]:
    """Linearize multiple paths in one call.

    Args:
        dag: DAG with sequence-annotated nodes.
        paths: List of paths (each a list of node names).

    Returns:
        List of DNA strings, one per path.
    """
    sequences: List[str] = []
    for idx, path in enumerate(paths):
        try:
            seq = linearize_path(dag, path)
            sequences.append(seq)
        except KeyError as exc:
            logger.error("Path %d: %s — skipping", idx, exc)
    return sequences


# ---------------------------------------------------------------------------
# Encoded sequence container
# ---------------------------------------------------------------------------

@dataclass
class EncodedSequence:
    """Holds all tensors / lists needed for a single model input.

    Attributes:
        token_ids: Integer token IDs (including special tokens and padding).
        tokens: Human-readable token strings.
        position_ids: Positional indices (0, 1, 2, …).
        attention_mask: 1 for real tokens, 0 for padding.
        original_sequence: The raw DNA string before tokenization.
    """
    token_ids: List[int]
    tokens: List[str]
    position_ids: List[int]
    attention_mask: List[int]
    original_sequence: str = ""


# ---------------------------------------------------------------------------
# Encoding pipeline
# ---------------------------------------------------------------------------

def encode_tokens(
    tokens: List[str],
    vocab: Dict[str, int],
    max_length: int = 128,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> EncodedSequence:
    """Wrap a token list with special tokens, pad/truncate, and create IDs.

    Layout: ``[CLS] tok1 tok2 … tokN [SEP] [PAD] [PAD] …``

    The sequence is truncated so that the total length (including [CLS] and
    [SEP]) does not exceed *max_length*.

    Args:
        tokens: Raw k-mer tokens produced by the tokenizer.
        vocab: Mapping from token string to integer ID.
        max_length: Desired fixed output length.
        cls_token: The classification token string.
        sep_token: The separator token string.
        pad_token: The padding token string.

    Returns:
        A fully populated :class:`EncodedSequence`.
    """
    # Reserve 2 positions for [CLS] and [SEP]
    max_content = max_length - 2
    truncated = tokens[:max_content]

    full_tokens = [cls_token] + truncated + [sep_token]
    pad_count = max_length - len(full_tokens)
    full_tokens += [pad_token] * pad_count

    # Unknown-token fallback
    unk_id = vocab.get("[UNK]", 0)
    token_ids = [vocab.get(t, unk_id) for t in full_tokens]

    attention_mask = [1] * (len(truncated) + 2) + [0] * pad_count
    position_ids = list(range(max_length))

    return EncodedSequence(
        token_ids=token_ids,
        tokens=full_tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
