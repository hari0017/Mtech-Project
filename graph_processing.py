"""
graph_processing.py — GFA/rGFA parsing and pangenome graph construction.

Reads GFA v1 (and a pragmatic subset of rGFA) files, builds a NetworkX
directed graph, and converts it into a DAG via strongly-connected-component
(SCC) contraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from utils import validate_dna_sequence, validate_file_exists

logger = logging.getLogger("pangenome_pipeline")


# ---------------------------------------------------------------------------
# Data containers for GFA records
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A GFA segment (S-line): a node carrying a DNA sequence.

    Attributes:
        name: Segment identifier (e.g. ``s1``).
        sequence: DNA string attached to the segment.
        tags: Optional tag dictionary parsed from extra GFA columns.
    """
    name: str
    sequence: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Link:
    """A GFA link (L-line): a directed edge between two oriented segments.

    Attributes:
        from_segment: Source segment name.
        from_orient: Orientation of the source (``+`` or ``-``).
        to_segment: Target segment name.
        to_orient: Orientation of the target.
        overlap: CIGAR overlap string (e.g. ``0M``).
    """
    from_segment: str
    from_orient: str
    to_segment: str
    to_orient: str
    overlap: str


@dataclass
class GFAGraph:
    """In-memory representation of a parsed GFA file.

    Attributes:
        segments: Mapping from segment name to :class:`Segment`.
        links: List of all :class:`Link` records.
        paths: Named paths (P-lines) mapping path name → ordered segment list.
        header: Raw header fields, if present.
    """
    segments: Dict[str, Segment] = field(default_factory=dict)
    links: List[Link] = field(default_factory=list)
    paths: Dict[str, List[str]] = field(default_factory=dict)
    header: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GFA Parsing
# ---------------------------------------------------------------------------

def parse_gfa(filepath: str) -> GFAGraph:
    """Parse a GFA v1 / rGFA file and return a :class:`GFAGraph`.

    Supports record types **H** (header), **S** (segment), **L** (link),
    and **P** (path).  Unknown line types are silently skipped.

    Args:
        filepath: Path to the ``.gfa`` file.

    Returns:
        Populated :class:`GFAGraph` instance.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If a segment contains invalid DNA characters.
    """
    validate_file_exists(filepath)
    graph = GFAGraph()

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            record_type = fields[0]

            if record_type == "H":
                _parse_header(fields, graph)
            elif record_type == "S":
                _parse_segment(fields, graph, line_no)
            elif record_type == "L":
                _parse_link(fields, graph, line_no)
            elif record_type == "P":
                _parse_path(fields, graph, line_no)
            else:
                logger.debug("Line %d: skipping unknown record type '%s'", line_no, record_type)

    logger.info(
        "Parsed GFA: %d segments, %d links, %d named paths",
        len(graph.segments), len(graph.links), len(graph.paths),
    )
    return graph


def _parse_header(fields: List[str], graph: GFAGraph) -> None:
    """Extract optional header tags (H-line)."""
    for tag in fields[1:]:
        parts = tag.split(":", maxsplit=2)
        if len(parts) == 3:
            graph.header[parts[0]] = parts[2]


def _parse_segment(fields: List[str], graph: GFAGraph, line_no: int) -> None:
    """Parse a segment line and add it to *graph*.

    Expected format: ``S  <name>  <sequence>  [tags…]``
    """
    if len(fields) < 3:
        logger.warning("Line %d: malformed S-line (too few fields), skipping", line_no)
        return

    name = fields[1]
    sequence = fields[2].upper()

    # Accept '*' as a placeholder for missing sequence data
    if sequence != "*" and not validate_dna_sequence(sequence):
        raise ValueError(
            f"Line {line_no}: segment '{name}' contains invalid DNA characters: "
            f"{sequence[:40]}{'…' if len(sequence) > 40 else ''}"
        )

    tags: Dict[str, str] = {}
    for tag in fields[3:]:
        parts = tag.split(":", maxsplit=2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]

    graph.segments[name] = Segment(name=name, sequence=sequence, tags=tags)


def _parse_link(fields: List[str], graph: GFAGraph, line_no: int) -> None:
    """Parse a link line and add it to *graph*.

    Expected format: ``L  <from>  <from_orient>  <to>  <to_orient>  <overlap>``
    """
    if len(fields) < 6:
        logger.warning("Line %d: malformed L-line (too few fields), skipping", line_no)
        return

    link = Link(
        from_segment=fields[1],
        from_orient=fields[2],
        to_segment=fields[3],
        to_orient=fields[4],
        overlap=fields[5],
    )
    graph.links.append(link)


def _parse_path(fields: List[str], graph: GFAGraph, line_no: int) -> None:
    """Parse a path line (P-line).

    Expected format: ``P  <path_name>  <seg1±,seg2±,…>  <overlaps>``
    """
    if len(fields) < 3:
        logger.warning("Line %d: malformed P-line (too few fields), skipping", line_no)
        return

    path_name = fields[1]
    segment_list = [seg.rstrip("+-") for seg in fields[2].split(",")]
    graph.paths[path_name] = segment_list


# ---------------------------------------------------------------------------
# NetworkX graph construction
# ---------------------------------------------------------------------------

def build_directed_graph(gfa: GFAGraph) -> nx.DiGraph:
    """Convert a :class:`GFAGraph` into a NetworkX :class:`~nx.DiGraph`.

    Each segment becomes a node whose ``sequence`` attribute stores its DNA
    string.  Each link becomes a directed edge whose ``overlap`` attribute
    stores the CIGAR string.

    Args:
        gfa: Parsed GFA data.

    Returns:
        A :class:`nx.DiGraph` with segment metadata on nodes.
    """
    G = nx.DiGraph()

    for seg in gfa.segments.values():
        G.add_node(seg.name, sequence=seg.sequence, tags=seg.tags)

    for link in gfa.links:
        G.add_edge(
            link.from_segment,
            link.to_segment,
            from_orient=link.from_orient,
            to_orient=link.to_orient,
            overlap=link.overlap,
        )

    logger.info(
        "Built directed graph: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Cycle removal via SCC contraction
# ---------------------------------------------------------------------------

def remove_cycles_via_scc(G: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """Remove cycles by contracting each SCC into a single super-node.

    Algorithm
    ---------
    1. Compute all strongly connected components (Tarjan / Kosaraju).
    2. Each SCC with more than one node is *contracted*: a new super-node
       replaces the entire SCC.  Its ``sequence`` is the concatenation of
       member sequences in sorted order (deterministic).
    3. Edges that previously connected nodes in different SCCs are rewired
       to the corresponding super-nodes.
    4. Self-loops introduced during contraction are removed.

    Args:
        G: A directed graph that may contain cycles.

    Returns:
        A 2-tuple ``(dag, scc_map)`` where

        * *dag* is a :class:`nx.DiGraph` guaranteed to be acyclic.
        * *scc_map* maps each (super-)node name in *dag* to the list of
          original node names it represents.
    """
    sccs: List[Set[str]] = list(nx.strongly_connected_components(G))
    logger.info("Found %d strongly connected components", len(sccs))

    # Mapping: original node → representative super-node name
    node_to_super: Dict[str, str] = {}
    scc_map: Dict[str, List[str]] = {}

    dag = nx.DiGraph()

    for scc in sccs:
        members = sorted(scc)  # deterministic ordering

        if len(members) == 1:
            # Trivial SCC (single node, possibly with a self-loop)
            node = members[0]
            super_name = node
            scc_map[super_name] = members
            node_to_super[node] = super_name
            dag.add_node(
                super_name,
                sequence=G.nodes[node].get("sequence", ""),
                tags=G.nodes[node].get("tags", {}),
                scc_members=members,
            )
        else:
            # Non-trivial SCC — merge into a super-node
            super_name = "SCC_" + "_".join(members)
            merged_seq = "".join(
                G.nodes[m].get("sequence", "") for m in members
            )
            merged_tags: Dict[str, str] = {}
            for m in members:
                merged_tags.update(G.nodes[m].get("tags", {}))

            scc_map[super_name] = members
            for m in members:
                node_to_super[m] = super_name

            dag.add_node(
                super_name,
                sequence=merged_seq,
                tags=merged_tags,
                scc_members=members,
            )
            logger.info(
                "Contracted SCC with %d nodes into super-node '%s'",
                len(members), super_name,
            )

    # Rewire edges
    for u, v, data in G.edges(data=True):
        su = node_to_super[u]
        sv = node_to_super[v]
        if su != sv and not dag.has_edge(su, sv):
            dag.add_edge(su, sv, **data)

    # Safety: remove any residual self-loops
    dag.remove_edges_from(nx.selfloop_edges(dag))

    assert nx.is_directed_acyclic_graph(dag), (
        "BUG: graph is still cyclic after SCC contraction"
    )
    logger.info(
        "DAG after SCC contraction: %d nodes, %d edges",
        dag.number_of_nodes(), dag.number_of_edges(),
    )
    return dag, scc_map


def process_graph(filepath: str) -> Tuple[nx.DiGraph, GFAGraph, Dict[str, List[str]]]:
    """End-to-end graph processing: parse → build → DAG.

    Convenience wrapper that chains :func:`parse_gfa`,
    :func:`build_directed_graph`, and :func:`remove_cycles_via_scc`.

    Args:
        filepath: Path to the GFA file.

    Returns:
        A 3-tuple ``(dag, gfa, scc_map)``.
    """
    gfa = parse_gfa(filepath)
    if not gfa.segments:
        raise ValueError(f"No segments found in '{filepath}'. Is the file valid GFA?")

    G = build_directed_graph(gfa)
    dag, scc_map = remove_cycles_via_scc(G)
    return dag, gfa, scc_map
