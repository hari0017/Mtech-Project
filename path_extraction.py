"""
path_extraction.py — Extract linear paths from a DAG.

Provides two complementary strategies:

1. **Topological traversal** — enumerate all source-to-sink paths in the DAG
   (with a configurable cap to avoid combinatorial explosion).
2. **Random walk** — sample paths by randomly choosing among successors,
   starting from source nodes.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger("pangenome_pipeline")


# ---------------------------------------------------------------------------
# Identify source / sink nodes
# ---------------------------------------------------------------------------

def _source_nodes(dag: nx.DiGraph) -> List[str]:
    """Return all nodes with in-degree 0 (entry points)."""
    return [n for n in dag.nodes() if dag.in_degree(n) == 0]


def _sink_nodes(dag: nx.DiGraph) -> List[str]:
    """Return all nodes with out-degree 0 (exit points)."""
    return [n for n in dag.nodes() if dag.out_degree(n) == 0]


# ---------------------------------------------------------------------------
# Topological path extraction
# ---------------------------------------------------------------------------

def extract_topological_paths(
    dag: nx.DiGraph,
    max_paths: int = 10,
) -> List[List[str]]:
    """Enumerate source-to-sink paths in topological order.

    Uses depth-first search from every source node.  To keep runtime
    manageable on large graphs the search stops after *max_paths* complete
    paths have been collected.

    Args:
        dag: A directed acyclic graph.
        max_paths: Maximum number of paths to return.

    Returns:
        A list of paths, where each path is an ordered list of node names.
    """
    if dag.number_of_nodes() == 0:
        logger.warning("DAG is empty — no paths to extract")
        return []

    sources = _source_nodes(dag)
    sinks = set(_sink_nodes(dag))

    if not sources:
        logger.warning("No source nodes found (all nodes have incoming edges)")
        return []
    if not sinks:
        logger.warning("No sink nodes found (all nodes have outgoing edges)")
        return []

    logger.info(
        "Extracting topological paths: %d sources, %d sinks, max %d paths",
        len(sources), len(sinks), max_paths,
    )

    paths: List[List[str]] = []

    for src in sources:
        if len(paths) >= max_paths:
            break
        # Iterative DFS using an explicit stack to avoid RecursionError
        # on large graphs. Each stack entry is (node, iterator_over_successors, path).
        stack: List[tuple] = [(src, iter(dag.successors(src)), [src])]

        # Check if source is also a sink
        if src in sinks:
            paths.append([src])
            if len(paths) >= max_paths:
                break

        while stack and len(paths) < max_paths:
            node, children, current_path = stack[-1]
            try:
                child = next(children)
                new_path = current_path + [child]
                if child in sinks:
                    paths.append(new_path)
                else:
                    stack.append((child, iter(dag.successors(child)), new_path))
            except StopIteration:
                stack.pop()

    logger.info("Extracted %d topological paths", len(paths))
    return paths


# ---------------------------------------------------------------------------
# Random walk path extraction
# ---------------------------------------------------------------------------

def extract_random_walk_paths(
    dag: nx.DiGraph,
    num_walks: int = 3,
    seed: Optional[int] = None,
) -> List[List[str]]:
    """Sample paths by performing random walks from source nodes.

    At each node the walker uniformly picks one of the successors.  The walk
    ends when a sink node (out-degree 0) is reached.

    Args:
        dag: A directed acyclic graph.
        num_walks: Number of random walks to perform.
        seed: Random seed for reproducibility.

    Returns:
        A list of sampled paths (may contain duplicates).
    """
    if dag.number_of_nodes() == 0:
        logger.warning("DAG is empty — no random walks possible")
        return []

    rng = random.Random(seed)
    sources = _source_nodes(dag)

    if not sources:
        logger.warning("No source nodes for random walk")
        return []

    paths: List[List[str]] = []

    for walk_idx in range(num_walks):
        start = rng.choice(sources)
        path = [start]
        current = start

        # Guard against infinite loops (should not happen in a DAG)
        max_steps = dag.number_of_nodes()
        steps = 0

        while True:
            successors = list(dag.successors(current))
            if not successors:
                break
            current = rng.choice(successors)
            path.append(current)
            steps += 1
            if steps > max_steps:
                logger.error(
                    "Random walk %d exceeded max steps — possible cycle in DAG",
                    walk_idx,
                )
                break

        paths.append(path)

    logger.info("Generated %d random-walk paths", len(paths))
    return paths


# ---------------------------------------------------------------------------
# Unified extraction entry point
# ---------------------------------------------------------------------------

def extract_paths(
    dag: nx.DiGraph,
    max_paths: int = 10,
    random_walk_paths: int = 3,
    random_seed: Optional[int] = None,
) -> List[List[str]]:
    """Extract paths from *dag* using both topological and random-walk strategies.

    Duplicate paths (same ordered node sequence) are removed.

    Args:
        dag: Directed acyclic graph.
        max_paths: Cap for topological enumeration.
        random_walk_paths: Number of random walks to sample.
        random_seed: Seed for the random-walk RNG.

    Returns:
        De-duplicated list of paths (each path is a list of node names).
    """
    topo_paths = extract_topological_paths(dag, max_paths=max_paths)
    rw_paths = extract_random_walk_paths(dag, num_walks=random_walk_paths, seed=random_seed)

    # De-duplicate while preserving order
    seen: Set[tuple] = set()
    unique_paths: List[List[str]] = []
    for p in topo_paths + rw_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    logger.info(
        "Total unique paths: %d (topo=%d, random-walk=%d, after dedup)",
        len(unique_paths), len(topo_paths), len(rw_paths),
    )
    return unique_paths
