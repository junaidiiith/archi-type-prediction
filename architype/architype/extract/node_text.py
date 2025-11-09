"""
Node-centric textual representations used for masked classification tasks.
Based on the original ``lang2graph.common`` helpers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import networkx as nx

from ..langgraph import LangGraph

from ..dataset.metadata import GraphMetadata
from .special_tokens import EDGE_END, EDGE_START, NODE_BEGIN, NODE_END
from ..utils.config import CONTAINMENT, REFERENCE, SUPERTYPE
from .tokenization import doc_tokenizer
from ..utils import get_node_neighbours


def format_path(
    graph: nx.DiGraph,
    path: Sequence[str],
    metadata: GraphMetadata,
    *,
    use_node_attributes: bool = False,
    use_node_types: bool = False,
    use_edge_label: bool = False,
    use_edge_types: bool = False,
    use_special_tokens: bool = False,
    no_labels: bool = False,
    preprocessor: Callable[[str], str] = doc_tokenizer,
    neg_sample: bool = False,
) -> str:
    """Format a traversal path into its textual representation."""

    node_cls_labels = metadata.node_cls
    edge_cls_labels = metadata.edge_cls
    
    def get_node_label(node: str) -> str:
        masked = graph.nodes[node].get("masked")
        
        def nt(g: LangGraph, n: str, lb: str):
            v = g.nodes[n].get(lb, '')
            if isinstance(v, bool):
                return lb.title() if v else ""
            return v
        
        node_types = ", ".join([t for t in [(
            f"{nt(graph, node, node_cls_label)}"
            if use_node_types and not masked and node_cls_label
            else ""
        ) for node_cls_label in node_cls_labels if node_cls_labels] if t]) if not masked else ""

        node_label = ""
        if not no_labels:
            node_label = get_node_name(
                graph.nodes[node],
                metadata.node_label,
                use_node_attributes,
                metadata.node_attributes,
            )

        if preprocessor:
            node_label = preprocessor(node_label)

        node_label = f"{node_label}({node_types})".strip() if node_label else node_types.strip()
        if use_special_tokens:
            node_label = f"{NODE_BEGIN}{node_label}{NODE_END}"

        return node_label.strip()

    def get_edge_label(n1: str, n2: str) -> str:
        edge_data = graph.get_edge_data(n1, n2)
        masked = edge_data.get("masked")
        edge_label = (
            edge_data.get(metadata.edge_label, "")
            if use_edge_label and not no_labels
            else ""
        )
        edge_types = ", ".join([t for t in [(
            f"{get_edge_data(edge_data, edge_cls_label, metadata.type)}"
            if use_edge_types and not masked and edge_cls_label
            else ""
        ) for edge_cls_label in edge_cls_labels if edge_cls_labels] if t]) if not masked else ""

        if preprocessor:
            edge_label = preprocessor(edge_label)

        edge_label = f"{edge_label}({edge_types})".strip() if edge_label else edge_types.strip()

        if use_special_tokens:
            edge_label = f"{EDGE_START}{edge_label}{EDGE_END}"

        return edge_label.strip()

    assert len(path) > 0, "Path must contain at least one node."
    formatted: List[str] = []
    for i in range(1, len(path)):
        n1 = path[i - 1]
        n2 = path[i]

        if not neg_sample:
            formatted.append(f"<{get_edge_label(n1, n2)}>")
        formatted.append(get_node_label(n2))

    node_str = get_node_label(path[0])
    if formatted:
        node_str += " " + " ".join(formatted).strip()

    return node_str


def get_edge_data(edge_data: Dict, edge_label: str, model_type: str):
    if model_type == "archimate":
        return edge_data.get(edge_label)
    if model_type == "ecore":
        return get_uml_edge_data(edge_data, edge_label)
    if model_type == "ontouml":
        return get_ontouml_edge_data(edge_data, edge_label)
    raise ValueError(f"Unknown edge label mapping for model type: {model_type}")


def get_uml_edge_data(edge_data: Dict, edge_label: str):
    if edge_label == "type":
        edge_type = edge_data.get("type")
        if edge_type == SUPERTYPE:
            return SUPERTYPE
        if edge_type == CONTAINMENT:
            return CONTAINMENT
        return REFERENCE
    if edge_label in edge_data:
        return edge_data[edge_label]
    raise ValueError(f"Unknown edge label: {edge_label}")


def get_ontouml_edge_data(edge_data: Dict, edge_label: str):
    try:
        return {"rel": "relates", "gen": "generalizes"}[edge_data.get(edge_label)]
    except KeyError as exc:  # pragma: no cover - awaiting data coverage
        raise ValueError(f"Unknown edge label: {edge_label}") from exc


def get_attribute_labels(node_data: Dict, attribute_labels: str) -> str:
    if isinstance(node_data[attribute_labels], list):
        if not node_data[attribute_labels]:
            return ""
        if isinstance(node_data[attribute_labels][0], str):
            return ", ".join(node_data[attribute_labels])
        if isinstance(node_data[attribute_labels][0], tuple):
            return ", ".join(f"{k}: {v}" for k, v in node_data[attribute_labels])
        if isinstance(node_data[attribute_labels][0], dict):
            return ", ".join(
                f"{k}: {v}" for d in node_data[attribute_labels] for k, v in d.items()
            )
        return ", ".join(node_data[attribute_labels])
    if isinstance(node_data[attribute_labels], dict):
        return ", ".join(f"{k}: {v}" for k, v in node_data[attribute_labels].items())
    return node_data[attribute_labels]


def get_node_name(
    node_data: Dict,
    label: Optional[str],
    use_attributes: bool,
    attribute_labels: Optional[str],
) -> str:
    if use_attributes and attribute_labels in node_data:
        attributes_str = "(" + get_attribute_labels(node_data, attribute_labels) + ")"
    else:
        attributes_str = ""

    node_label = node_data.get(label, "") if label else ""
    node_label = (
        "" if node_label and node_label.lower() in ["null", "none"] else node_label
    )
    return f"{node_label}{attributes_str}".strip()


def get_node_data(node_data: Dict, node_label: str, model_type: str):
    if model_type == "archimate":
        return node_data.get(node_label)
    if model_type == "ecore":
        return node_data.get(node_label, "")
    if model_type == "ontouml":
        return node_data.get(node_label, "")
    raise ValueError(f"Unknown model type: {model_type}")


def get_node_text(
    graph: nx.DiGraph,
    node: str,
    d: int,
    metadata: GraphMetadata,
    *,
    use_node_attributes: bool = False,
    use_node_types: bool = False,
    use_edge_types: bool = False,
    use_edge_label: bool = False,
    use_special_tokens: bool = False,
    no_labels: bool = False,
    preprocessor: Callable[[str], str] = doc_tokenizer,
    exclude_edges: Optional[List] = None,
) -> str:
    node_cls_labels = metadata.node_cls
    masked = graph.nodes[node].get("masked")
    graph.nodes[node]["masked"] = True if not exclude_edges else False

    node_neighbour_texts: List[str] = []
    node_neighbours = get_node_neighbours(graph, node, d, exclude_edges=exclude_edges)
    for neighbour in node_neighbours:
        unique_paths = list(nx.all_simple_paths(graph, node, neighbour, cutoff=d))
        node_neighbour_texts.extend(
            [
                format_path(
                    graph=graph,
                    path=path,
                    metadata=metadata,
                    
                    use_node_attributes=use_node_attributes,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    use_edge_label=use_edge_label,
                    use_special_tokens=use_special_tokens,
                    no_labels=no_labels,
                    
                    preprocessor=preprocessor,
                    neg_sample=False,
                )
                for path in unique_paths
            ]
        )

    graph.nodes[node]["masked"] = masked or False
    node_str = "\n".join(node_neighbour_texts).strip() if node_neighbour_texts else ""

    if "stereotype" in node_cls_labels:
        node_str = graph.nodes[node]["type"].title() + " " + node_str

    return node_str.strip()


def get_node_texts(
    graph: nx.DiGraph,
    d: int,
    metadata: GraphMetadata,
    *,
    use_node_attributes: bool = False,
    use_node_types: bool = False,
    use_edge_types: bool = False,
    use_edge_label: bool = False,
    use_special_tokens: bool = False,
    no_labels: bool = False,
    preprocessor: Callable[[str], str] = doc_tokenizer,
) -> Dict[str, str]:
    
    
    paths_dict: Dict[str, str] = {}
    for node in graph.nodes:
        paths_dict[node] = get_node_text(
            graph=graph,
            node=node,
            d=d,
            metadata=metadata,
            
            use_node_attributes=use_node_attributes,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_edge_label=use_edge_label,
            use_special_tokens=use_special_tokens,
            no_labels=no_labels,
            preprocessor=preprocessor,
        )
    return paths_dict



__all__ = ["get_node_texts", "get_node_text"]
