"""
tokenizer_module.py — K-mer tokenization for DNA sequences.

Provides a sliding-window k-mer tokenizer and a vocabulary builder.
The module is structured so that a BPE tokenizer (e.g. from HuggingFace
``tokenizers``) can be plugged in as a drop-in replacement via the
abstract :class:`BaseTokenizer` interface.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger("pangenome_pipeline")

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseTokenizer(ABC):
    """Interface that every tokenizer in this project must implement."""

    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        """Split a DNA sequence into tokens."""

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return the token → ID mapping."""


# ---------------------------------------------------------------------------
# K-mer tokenizer
# ---------------------------------------------------------------------------

class KmerTokenizer(BaseTokenizer):
    """Sliding-window k-mer tokenizer for DNA strings.

    Example (k=3)::

        >>> tok = KmerTokenizer(k=3)
        >>> tok.tokenize("ATCGATCG")
        ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']

    The vocabulary is built exhaustively over the DNA alphabet ``{A, C, G, T}``
    so that every possible k-mer has a stable integer ID.

    Args:
        k: Length of each k-mer. Common values are 3 and 6.
        stride: Step size of the sliding window (default 1).
    """

    def __init__(self, k: int = 3, stride: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        if stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {stride}")

        self.k = k
        self.stride = stride
        self._vocab = self._build_vocab()
        logger.info(
            "KmerTokenizer initialised: k=%d, stride=%d, vocab_size=%d",
            k, stride, len(self._vocab),
        )

    @classmethod
    def from_pretrained(cls, model_name: str, k: int = 3):
        """Use the checkpoint's token IDs instead of a generated ID ordering."""
        from transformers import BertTokenizer

        checkpoint_tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer = cls(k=k)
        vocab = checkpoint_tokenizer.get_vocab()
        if not set(tokenizer._vocab).issubset(vocab):
            raise ValueError(
                f"Checkpoint tokenizer does not contain the complete {k}-mer vocabulary "
                "and required special tokens. Match --kmer to --model."
            )
        tokenizer._vocab = dict(vocab)
        tokenizer.checkpoint_tokenizer = checkpoint_tokenizer
        return tokenizer

    def save_pretrained(self, output_dir: str) -> None:
        self.checkpoint_tokenizer.save_pretrained(output_dir)

    # ---- vocabulary construction ----

    def _build_vocab(self) -> Dict[str, int]:
        """Generate the full k-mer vocabulary plus special tokens.

        The vocabulary is deterministic: special tokens come first, then all
        k-mers in lexicographic order.
        """
        vocab: Dict[str, int] = {}
        idx = 0

        # Special tokens first
        for tok in SPECIAL_TOKENS:
            vocab[tok] = idx
            idx += 1

        # All possible k-mers (4^k entries)
        bases = "ACGT"
        for kmer_tuple in itertools.product(bases, repeat=self.k):
            kmer = "".join(kmer_tuple)
            vocab[kmer] = idx
            idx += 1

        return vocab

    # ---- public API ----

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize *sequence* into overlapping k-mers.

        Characters not in {A, C, G, T} are converted to uppercase first;
        any remaining non-ACGT characters cause the enclosing k-mer to be
        tagged as ``[UNK]``.

        Args:
            sequence: A raw DNA string.

        Returns:
            Ordered list of k-mer token strings.
        """
        sequence = sequence.upper()
        if len(sequence) < self.k:
            logger.warning(
                "Sequence length (%d) < k (%d) — returning empty token list",
                len(sequence), self.k,
            )
            return []

        tokens: List[str] = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i : i + self.k]
            if kmer in self._vocab:
                tokens.append(kmer)
            else:
                tokens.append("[UNK]")
        return tokens

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# BPE tokenizer stub (future extension)
# ---------------------------------------------------------------------------

class BPETokenizerWrapper(BaseTokenizer):
    """Placeholder for a HuggingFace ``tokenizers.Tokenizer`` based on BPE.

    This class shows how a BPE tokenizer can be plugged into the same
    interface.  To activate it, train or load a BPE model and pass the
    path here.

    Example usage (once a trained tokenizer is available)::

        from tokenizers import Tokenizer as HFTokenizer
        bpe = BPETokenizerWrapper("/path/to/bpe_tokenizer.json")
        tokens = bpe.tokenize("ATCGATCG")
    """

    def __init__(self, tokenizer_path: Optional[str] = None) -> None:
        self._tokenizer = None
        self._vocab: Dict[str, int] = {}

        if tokenizer_path is not None:
            try:
                from tokenizers import Tokenizer as HFTokenizer  # type: ignore
                self._tokenizer = HFTokenizer.from_file(tokenizer_path)
                self._vocab = self._tokenizer.get_vocab()
                logger.info(
                    "Loaded BPE tokenizer from '%s' (vocab size %d)",
                    tokenizer_path, len(self._vocab),
                )
            except ImportError:
                raise ImportError(
                    "The `tokenizers` package is required for BPE. "
                    "Install it with: pip install tokenizers"
                )
        else:
            logger.info("BPETokenizerWrapper created without a model (stub mode)")

    def tokenize(self, sequence: str) -> List[str]:
        if self._tokenizer is None:
            raise RuntimeError(
                "BPE tokenizer not loaded. Provide a valid tokenizer_path."
            )
        encoding = self._tokenizer.encode(sequence.upper())
        return encoding.tokens

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)
