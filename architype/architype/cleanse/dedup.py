"""
Duplicate detection helpers for LangGraph datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import networkx as nx


EdgeSignature = str


def _normalize_type(raw: object) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, (list, tuple, set)):
        return "|".join(str(item).strip().lower() for item in raw if item)
    return str(raw).strip().lower()


def _serialize_edge(graph: nx.DiGraph, edge: Tuple[str, str]) -> EdgeSignature:
    """
    Serialize an edge using the format ``SrcName(SrcType) edge_type TgtName(TgtType)``.
    """

    src, tgt = edge
    src_data = graph.nodes[src]
    tgt_data = graph.nodes[tgt]
    edge_data = graph.edges[edge]

    src_name = src_data.get("name") or str(src)
    tgt_name = tgt_data.get("name") or str(tgt)
    src_type = _normalize_type(src_data.get("type") or src_data.get("cls"))
    tgt_type = _normalize_type(tgt_data.get("type") or tgt_data.get("cls"))
    edge_type = _normalize_type(
        edge_data.get("type") or edge_data.get("cls") or "relates-to"
    )

    return f"{src_name}({src_type}) {edge_type} {tgt_name}({tgt_type})"


def build_edge_signature(graph: nx.DiGraph) -> Set[EdgeSignature]:
    """
    Construct the canonical edge signature set for a graph.
    """

    return {_serialize_edge(graph, edge) for edge in graph.edges}


@dataclass
class DuplicateRecord:
    source_index: int
    duplicate_index: int
    overlap_ratio: float
    shared_edges: Set[EdgeSignature]


def deduplicate_graphs(
    graphs: Sequence[nx.DiGraph],
    *,
    edge_overlap_threshold: float = 0.8,
) -> Tuple[List[nx.DiGraph], List[DuplicateRecord]]:
    """
    Remove duplicate graphs based on edge overlap.

    Two models are considered duplicates when the proportion of shared serialized
    edges is greater than or equal to ``edge_overlap_threshold`` relative to the
    smaller graph.
    """

    unique_graphs: List[nx.DiGraph] = []
    signature_cache: List[Set[EdgeSignature]] = []
    duplicates: List[DuplicateRecord] = []

    for idx, graph in enumerate(graphs):
        signature = build_edge_signature(graph)
        if not signature:
            unique_graphs.append(graph)
            signature_cache.append(signature)
            continue

        found_duplicate = False
        for unique_idx, unique_signature in enumerate(signature_cache):
            if not unique_signature:
                continue
            shared_edges = signature & unique_signature
            baseline = min(len(signature), len(unique_signature))
            overlap_ratio = len(shared_edges) / baseline if baseline else 0.0
            if overlap_ratio >= edge_overlap_threshold:
                duplicates.append(
                    DuplicateRecord(
                        source_index=unique_idx,
                        duplicate_index=idx,
                        overlap_ratio=overlap_ratio,
                        shared_edges=shared_edges,
                    )
                )
                found_duplicate = True
                break

        if not found_duplicate:
            unique_graphs.append(graph)
            signature_cache.append(signature)

    return unique_graphs, duplicates


__all__ = ["deduplicate_graphs", "build_edge_signature", "DuplicateRecord"]
