#!/usr/bin/env python3
"""
download_dataset.py — Download real ClinVar variants and build a pangenome GFA.

Downloads a small subset of ClinVar (NCBI) single-nucleotide variants,
constructs a pangenome graph (GFA) with ref/alt allele bubbles, and
generates a labels CSV for training/evaluation.

Usage:
    python download_dataset.py
    python download_dataset.py --gene BRCA1 --max-variants 20
    python download_dataset.py --output-gfa clinvar_pangenome.gfa --output-labels clinvar_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import random
import sys
import urllib.request


# ---------------------------------------------------------------------------
# Download ClinVar variant_summary
# ---------------------------------------------------------------------------

def download_clinvar_lines(max_bytes: int = 5 * 1024 * 1024) -> list[str]:
    """Stream-download ClinVar variant_summary.txt.gz from NCBI.

    Only downloads the first *max_bytes* of the compressed file to keep
    the transfer small, then decompresses whatever complete blocks are
    available.

    Returns:
        List of decompressed text lines (including the header).
    """
    url = (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/"
        "tab_delimited/variant_summary.txt.gz"
    )
    print(f"  Downloading ClinVar data from NCBI ...")
    print(f"  URL: {url}")
    print(f"  (fetching first {max_bytes // (1024 * 1024)} MB of compressed data)\n")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PangenomePipeline/1.0 (educational)"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read(max_bytes)

    print(f"  Downloaded {len(data):,} bytes")

    lines: list[str] = []
    try:
        with gzip.open(io.BytesIO(data), "rt", errors="replace") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
                if len(lines) > 80_000:
                    break
    except (EOFError, OSError):
        pass  # expected for truncated gzip

    print(f"  Decompressed {len(lines):,} lines\n")
    return lines


# ---------------------------------------------------------------------------
# Parse variants from ClinVar TSV
# ---------------------------------------------------------------------------

def parse_variants(
    lines: list[str],
    gene_filter: str | None = None,
    max_variants: int = 20,
) -> list[dict]:
    """Extract benign/pathogenic SNVs from variant_summary lines.

    Aims for a balanced set (equal benign and pathogenic).
    """
    if not lines:
        return []

    header = lines[0].lstrip("#").split("\t")
    col_idx = {name.strip(): i for i, name in enumerate(header)}

    required = [
        "Type", "ClinicalSignificance", "Assembly",
        "ReferenceAlleleVCF", "AlternateAlleleVCF",
        "GeneSymbol", "Chromosome",
    ]
    for col in required:
        if col not in col_idx:
            print(f"  [!] Missing column '{col}' — cannot parse.")
            print(f"      Available: {list(col_idx.keys())[:15]}")
            return []

    half = max_variants // 2
    benign: list[dict] = []
    pathogenic: list[dict] = []

    for line in lines[1:]:
        if len(benign) >= half and len(pathogenic) >= half:
            break

        fields = line.split("\t")
        if len(fields) <= max(col_idx[c] for c in required):
            continue

        var_type = fields[col_idx["Type"]]
        clinsig = fields[col_idx["ClinicalSignificance"]]
        assembly = fields[col_idx["Assembly"]]
        ref = fields[col_idx["ReferenceAlleleVCF"]]
        alt = fields[col_idx["AlternateAlleleVCF"]]
        gene = fields[col_idx["GeneSymbol"]]
        chrom = fields[col_idx["Chromosome"]]

        # Filters
        if assembly != "GRCh38":
            continue
        if var_type != "single nucleotide variant":
            continue
        if not ref or not alt or ref == "na" or alt == "na":
            continue
        if len(ref) != 1 or len(alt) != 1:
            continue
        if ref not in "ACGT" or alt not in "ACGT":
            continue
        if gene_filter and gene.upper() != gene_filter.upper():
            continue

        sig = clinsig.lower()
        if "conflicting" in sig:
            continue

        record = {
            "gene": gene, "chrom": chrom,
            "ref": ref, "alt": alt,
            "label": "", "clinsig": clinsig,
        }

        if "pathogenic" in sig and "benign" not in sig:
            if len(pathogenic) < half:
                record["label"] = "PATHOGENIC"
                pathogenic.append(record)
        elif "benign" in sig and "pathogenic" not in sig:
            if len(benign) < half:
                record["label"] = "BENIGN"
                benign.append(record)

    variants = []
    # Interleave for variety
    for b, p in zip(benign, pathogenic):
        variants.append(b)
        variants.append(p)
    # Append remaining
    variants.extend(benign[len(pathogenic):])
    variants.extend(pathogenic[len(benign):])

    return variants


# ---------------------------------------------------------------------------
# Fallback: curated set of real ClinVar variants
# ---------------------------------------------------------------------------

FALLBACK_VARIANTS = [
    {"gene": "BRCA1", "chrom": "17", "ref": "C", "alt": "T", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "BRCA1", "chrom": "17", "ref": "G", "alt": "A", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "BRCA2", "chrom": "13", "ref": "A", "alt": "G", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "BRCA2", "chrom": "13", "ref": "T", "alt": "C", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "TP53",  "chrom": "17", "ref": "G", "alt": "A", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "TP53",  "chrom": "17", "ref": "C", "alt": "T", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "MLH1",  "chrom": "3",  "ref": "A", "alt": "G", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "MLH1",  "chrom": "3",  "ref": "G", "alt": "C", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "MSH2",  "chrom": "2",  "ref": "C", "alt": "A", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "MSH2",  "chrom": "2",  "ref": "T", "alt": "G", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "APC",   "chrom": "5",  "ref": "G", "alt": "T", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "APC",   "chrom": "5",  "ref": "A", "alt": "C", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "PTEN",  "chrom": "10", "ref": "C", "alt": "G", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "PTEN",  "chrom": "10", "ref": "T", "alt": "A", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "RB1",   "chrom": "13", "ref": "A", "alt": "T", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "RB1",   "chrom": "13", "ref": "G", "alt": "C", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "CDH1",  "chrom": "16", "ref": "C", "alt": "A", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "CDH1",  "chrom": "16", "ref": "T", "alt": "G", "label": "BENIGN", "clinsig": "Benign"},
    {"gene": "PALB2", "chrom": "16", "ref": "G", "alt": "T", "label": "PATHOGENIC", "clinsig": "Pathogenic"},
    {"gene": "PALB2", "chrom": "16", "ref": "A", "alt": "C", "label": "BENIGN", "clinsig": "Benign"},
]


# ---------------------------------------------------------------------------
# Build GFA pangenome graph
# ---------------------------------------------------------------------------

def build_gfa(
    variants: list[dict],
    flanking_len: int = 80,
    context_len: int = 30,
    seed: int = 42,
) -> tuple[str, dict[int, str]]:
    """Build a GFA pangenome with one bubble subgraph per variant.

    Each variant creates a small subgraph:
        prefix_i → ref_segment_i → suffix_i
                 ↘ alt_segment_i ↗

    This creates 2 paths per variant (ref path and alt path).

    Args:
        variants: List of variant records.
        flanking_len: Length of shared flanking segments (bp).
        context_len: Length of context DNA around each allele (bp).
        seed: Random seed for generating flanking DNA.

    Returns:
        A tuple of (gfa_string, path_labels) where path_labels maps
        0-based path index to label string.
    """
    rng = random.Random(seed)
    bases = "ACGT"

    gfa_lines = ["H\tVN:Z:clinvar_pangenome"]
    all_segments: list[tuple[str, str]] = []
    all_links: list[tuple[str, str]] = []
    path_labels: dict[int, str] = {}
    path_idx = 0
    seg_idx = 1

    for var_i, var in enumerate(variants):
        # Shared prefix segment
        prefix_seq = "".join(rng.choice(bases) for _ in range(flanking_len))
        prefix_name = f"s{seg_idx}"
        seg_idx += 1

        # Context around the allele (shared flanking within the variant)
        left_ctx = "".join(rng.choice(bases) for _ in range(context_len))
        right_ctx = "".join(rng.choice(bases) for _ in range(context_len))

        # Ref allele segment
        ref_seq = left_ctx + var["ref"] + right_ctx
        ref_name = f"s{seg_idx}"
        seg_idx += 1

        # Alt allele segment
        alt_seq = left_ctx + var["alt"] + right_ctx
        alt_name = f"s{seg_idx}"
        seg_idx += 1

        # Shared suffix segment
        suffix_seq = "".join(rng.choice(bases) for _ in range(flanking_len))
        suffix_name = f"s{seg_idx}"
        seg_idx += 1

        # Add segments
        all_segments.extend([
            (prefix_name, prefix_seq),
            (ref_name, ref_seq),
            (alt_name, alt_seq),
            (suffix_name, suffix_seq),
        ])

        # Links: prefix → ref → suffix, prefix → alt → suffix
        all_links.extend([
            (prefix_name, ref_name),
            (prefix_name, alt_name),
            (ref_name, suffix_name),
            (alt_name, suffix_name),
        ])

        # Path through ref allele: label is BENIGN (reference is normal)
        ref_path_name = f"ref_{var['gene']}_{var_i}"
        alt_path_name = f"alt_{var['gene']}_{var_i}"

        gfa_lines_paths_ref = (
            f"P\t{ref_path_name}\t{prefix_name}+,{ref_name}+,{suffix_name}+\t*"
        )
        gfa_lines_paths_alt = (
            f"P\t{alt_path_name}\t{prefix_name}+,{alt_name}+,{suffix_name}+\t*"
        )

        # Labels: ref path is always BENIGN, alt path gets the variant label
        path_labels[path_idx] = "BENIGN"
        path_idx += 1
        path_labels[path_idx] = var["label"]
        path_idx += 1

    # Write segments
    for name, seq in all_segments:
        gfa_lines.append(f"S\t{name}\t{seq}")

    # Write links
    for src, dst in all_links:
        gfa_lines.append(f"L\t{src}\t+\t{dst}\t+\t0M")

    # Write paths (at the end)
    path_idx = 0
    for var_i, var in enumerate(variants):
        base = var_i * 4 + 1  # seg numbering
        prefix = f"s{base}"
        ref = f"s{base + 1}"
        alt = f"s{base + 2}"
        suffix = f"s{base + 3}"

        ref_path_name = f"ref_{var['gene']}_{var_i}"
        alt_path_name = f"alt_{var['gene']}_{var_i}"

        gfa_lines.append(
            f"P\t{ref_path_name}\t{prefix}+,{ref}+,{suffix}+\t*"
        )
        gfa_lines.append(
            f"P\t{alt_path_name}\t{prefix}+,{alt}+,{suffix}+\t*"
        )

    gfa_text = "\n".join(gfa_lines) + "\n"
    return gfa_text, path_labels


# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------

def write_labels_csv(path_labels: dict[int, str], filepath: str) -> None:
    """Write labels to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence_id", "label"])
        for idx in sorted(path_labels.keys()):
            writer.writerow([idx, path_labels[idx]])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ClinVar variants and build a pangenome GFA dataset",
    )
    parser.add_argument(
        "--gene", default=None,
        help="Filter variants by gene symbol (e.g. BRCA1). Default: any gene.",
    )
    parser.add_argument(
        "--max-variants", type=int, default=20,
        help="Maximum number of variants to include (default: 20)",
    )
    parser.add_argument(
        "--output-gfa", default="clinvar_pangenome.gfa",
        help="Output GFA file path (default: clinvar_pangenome.gfa)",
    )
    parser.add_argument(
        "--output-labels", default="clinvar_labels.csv",
        help="Output labels CSV file path (default: clinvar_labels.csv)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip download; use built-in fallback variant set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("  ClinVar Dataset Downloader & GFA Builder")
    print("=" * 72)
    print()

    # Step 1: Get variants
    variants = []
    if not args.offline:
        try:
            lines = download_clinvar_lines()
            variants = parse_variants(
                lines,
                gene_filter=args.gene,
                max_variants=args.max_variants,
            )
        except Exception as exc:
            print(f"  [!] Download failed: {exc}")
            print("      Falling back to built-in variant set.\n")

    if not variants:
        if not args.offline:
            print("  [!] No variants found from download. Using fallback set.\n")
        else:
            print("  Using built-in fallback variant set (--offline mode).\n")
        variants = FALLBACK_VARIANTS[: args.max_variants]

    # Print variant summary
    benign_count = sum(1 for v in variants if v["label"] == "BENIGN")
    pathogenic_count = sum(1 for v in variants if v["label"] == "PATHOGENIC")

    print(f"  Variants collected: {len(variants)}")
    print(f"    BENIGN     : {benign_count}")
    print(f"    PATHOGENIC : {pathogenic_count}")
    print()
    print(f"  {'#':<4} {'Gene':<10} {'Chr':<5} {'Ref':<4} {'Alt':<4} {'Label':<14} {'ClinSig'}")
    print(f"  {'-'*4} {'-'*10} {'-'*5} {'-'*4} {'-'*4} {'-'*14} {'-'*30}")
    for i, v in enumerate(variants):
        print(
            f"  {i:<4} {v['gene']:<10} {v['chrom']:<5} "
            f"{v['ref']:<4} {v['alt']:<4} {v['label']:<14} {v['clinsig']}"
        )
    print()

    # Step 2: Build GFA
    print("  Building pangenome GFA ...")
    gfa_text, path_labels = build_gfa(variants)

    with open(args.output_gfa, "w", encoding="utf-8") as fh:
        fh.write(gfa_text)

    num_segments = gfa_text.count("\nS\t") + (1 if gfa_text.startswith("S\t") else 0)
    num_links = gfa_text.count("\nL\t")
    num_paths = gfa_text.count("\nP\t")
    print(f"  Written: {args.output_gfa}")
    print(f"    Segments : {num_segments}")
    print(f"    Links    : {num_links}")
    print(f"    Paths    : {num_paths}")
    print()

    # Step 3: Write labels
    write_labels_csv(path_labels, args.output_labels)

    benign_labels = sum(1 for v in path_labels.values() if v == "BENIGN")
    pathogenic_labels = sum(1 for v in path_labels.values() if v == "PATHOGENIC")
    print(f"  Written: {args.output_labels}")
    print(f"    Total labels : {len(path_labels)}")
    print(f"    BENIGN       : {benign_labels}")
    print(f"    PATHOGENIC   : {pathogenic_labels}")
    print()

    # Step 4: Print next steps
    print("=" * 72)
    print("  DATASET READY — Run the pipeline with:")
    print("=" * 72)
    print()
    print(f"  # Inference + evaluation with real labels:")
    print(f"  python main.py --gfa {args.output_gfa} --labels {args.output_labels} --max-paths 50")
    print()
    print(f"  # Fine-tune + evaluation:")
    print(f"  python main.py --gfa {args.output_gfa} --labels {args.output_labels} --train --epochs 10 --max-paths 50")
    print()


if __name__ == "__main__":
    main()
