"""
Core LangGraph abstractions built on top of NetworkX. These definitions are
ported from the earlier codebase and will be extended as the modular pipeline
is implemented.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterable, Tuple
from uuid import uuid4

import networkx as nx
import numpy as np
import torch

from ..utils import md5_hash


class LangGraph(nx.DiGraph):
    """Directed graph with stable identifiers for nodes and edges."""

    def __init__(self) -> None:
        super().__init__()
        self.id = uuid4().hex
        self.node_label_to_id: Dict[Any, int] = {}
        self.id_to_node_label: Dict[int, Any] = {}
        self.edge_label_to_id: Dict[Any, int] = {}
        self.id_to_edge_label: Dict[int, Any] = {}

    @abstractmethod
    def create_graph(self) -> None:
        """Subclasses should populate the graph structure."""

    def set_numbered_labels(self) -> None:
        """Assign integer ids to nodes/edges for efficient tensor access."""
        self.node_label_to_id = {label: i for i, label in enumerate(self.nodes())}
        self.id_to_node_label = {i: label for i, label in enumerate(self.nodes())}

        self.edge_label_to_id = {label: i for i, label in enumerate(self.edges())}
        self.id_to_edge_label = {i: label for i, label in enumerate(self.edges())}

        self.numbered_graph = self.get_numbered_graph()
        self.edge_to_idx = {
            edge: idx for idx, edge in enumerate(self.numbered_graph.edges())
        }
        self.idx_to_edge = {
            idx: edge for idx, edge in enumerate(self.numbered_graph.edges())
        }

    def get_numbered_graph(self) -> nx.DiGraph:
        nodes: Iterable[Tuple[int, Dict[str, Any]]] = [
            (self.node_label_to_id[i], data) for i, data in list(self.nodes(data=True))
        ]
        edges: Iterable[Tuple[int, int, Dict[str, Any]]] = [
            (
                self.node_label_to_id[i],
                self.node_label_to_id[j],
                data,
            )
            for i, j, data in list(self.edges(data=True))
        ]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    @property
    def enr(self) -> float:
        if self.number_of_nodes() == 0:
            return -1
        return self.number_of_edges() / self.number_of_nodes()

    @property
    def edge_index(self) -> np.ndarray:
        edge_index = (
            torch.tensor(list(self.numbered_graph.edges)).t().contiguous().numpy()
        )
        return edge_index

    @property
    def hash(self) -> str:
        return md5_hash(str(sorted(self.edges)))

    def get_edge_id(self, edge: Tuple[Any, Any]) -> int:
        return self.edge_label_to_id[edge]

    def get_edge_label(self, edge_id: int) -> Any:
        return self.edge_label_to_id[edge_id]

    def get_node_id(self, node: Any) -> int:
        return self.node_label_to_id[node]

    def get_node_label(self, node_id: int) -> Any:
        return self.node_label_to_id[node_id]
    


def create_graph_from_edge_index(
    graph: LangGraph, edge_index: np.ndarray
) -> nx.DiGraph:
    """
    Create a subgraph from the provided edge indices.
    """

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(list(graph.numbered_graph.nodes(data=True)))
    subgraph.add_edges_from(
        [(u, v, graph.numbered_graph.edges[u, v]) for u, v in edge_index.T]
    )
    for node, _ in subgraph.nodes(data=True):
        data = graph.numbered_graph.nodes[node]
        subgraph.nodes[node].update(data)

    subgraph.node_label_to_id = graph.node_label_to_id
    subgraph.id_to_node_label = graph.id_to_node_label
    subgraph.edge_label_to_id = graph.edge_label_to_id
    subgraph.id_to_edge_label = graph.id_to_edge_label

    if len(edge_index) > 0:
        assert subgraph.number_of_edges() == edge_index.shape[1], (
            "Number of edges mismatch"
        )

    return subgraph


__all__ = ["LangGraph", "create_graph_from_edge_index"]
