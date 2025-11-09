"""
Utility helpers migrated from the previous codebase. These provide hashing and
graph traversal primitives used by the LangGraph extraction utilities. They
will be reorganized into dedicated modules (io, logging, seed management) in a
later step of the refactor.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


def md5_hash(input_string: str) -> str:
    digest = hashlib.md5()
    digest.update(input_string.encode("utf-8"))
    return digest.hexdigest()


def find_nodes_within_distance(
    graph: nx.DiGraph,
    start_node: Any,
    distance: int,
    exclude_edges: Optional[List[Tuple[Any, Any]]] = None,
) -> List[Tuple[Any, int]]:
    """
    Breadth-first traversal up to ``distance`` hops, optionally skipping edges.
    """

    excluded: Set[Tuple[Any, Any]] = set(exclude_edges or [])
    queue: deque[Tuple[Any, int]] = deque([(start_node, 0)])
    visited: Dict[Any, int] = {}

    while queue:
        node, dist = queue.popleft()
        if dist > distance:
            continue

        if node not in visited or dist < visited[node]:
            visited[node] = dist

        if dist < distance:
            for nbr in graph.neighbors(node):
                if nbr == node:
                    continue
                if (node, nbr) in excluded:
                    continue
                if nbr in visited and visited[nbr] <= dist + 1:
                    continue
                queue.append((nbr, dist + 1))

    return sorted(visited.items(), key=lambda x: x[1])


def get_node_neighbours(
    graph: nx.DiGraph,
    start_node: Any,
    distance: int,
    exclude_edges: Optional[Iterable[Tuple[Any, Any]]] = None,
) -> List[Any]:
    """
    Return the set of nodes exactly ``distance`` hops from ``start_node``.
    """

    neighbours = find_nodes_within_distance(
        graph, start_node, distance, list(exclude_edges or [])
    )
    if not neighbours:
        return []
    max_distance = max(d for _, d in neighbours)
    distance = min(distance, max_distance)
    return [node for node, d in neighbours if d == distance]


__all__ = ["md5_hash", "find_nodes_within_distance", "get_node_neighbours"]

