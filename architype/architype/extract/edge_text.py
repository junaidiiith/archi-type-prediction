"""
Edge-centric textual extraction utilities. Built on top of the node text
representation helpers.
"""

from __future__ import annotations

from typing import Callable

from ..dataset.metadata import GraphMetadata
from ..langgraph.base import LangGraph
from .node_text import (
    get_edge_data,
    get_node_text,
)
from .special_tokens import EDGE_END, EDGE_START
from ..utils.config import EDGE_CLS_TASK, LINK_PRED_TASK
from .tokenization import doc_tokenizer


def get_edge_texts(
    graph: LangGraph,
    edge: tuple,
    d: int,
    task_type: str,
    metadata: GraphMetadata,
    *,
    use_node_attributes: bool = False,
    use_node_types: bool = False,
    use_edge_types: bool = False,
    use_edge_label: bool = False,
    use_special_tokens: bool = False,
    no_labels: bool = False,
    preprocessor: Callable[[str], str] = doc_tokenizer,
    neg_samples: bool = False,
):
    edge_cls_labels = metadata.edge_cls
    n1, n2 = edge
    if not neg_samples:
        masked = graph.edges[n1, n2].get("masked")
        graph.edges[n1, n2]["masked"] = True
    else:
        masked = False

    n1_text = get_node_text(
        graph=graph,
        node=n1,
        d=d,
        metadata=metadata,
        use_node_attributes=use_node_attributes,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        use_edge_label=use_edge_label,
        use_special_tokens=use_special_tokens,
        no_labels=no_labels,
        preprocessor=preprocessor,
        exclude_edges=[edge],
    )
    n2_text = get_node_text(
        graph=graph,
        node=n2,
        d=d,
        metadata=metadata,
        use_node_attributes=use_node_attributes,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        use_edge_label=use_edge_label,
        use_special_tokens=use_special_tokens,
        no_labels=no_labels,
        preprocessor=preprocessor,
        exclude_edges=[edge],
    )

    edge_text = ""

    if not neg_samples:
        graph.edges[n1, n2]["masked"] = masked or False

        edge_data = graph.get_edge_data(n1, n2)
        edge_label = (
            edge_data.get(metadata.edge_label, "")
            if use_edge_label and not no_labels
            else ""
        )

        if task_type not in [EDGE_CLS_TASK, LINK_PRED_TASK]:
            if use_edge_label:
                edge_text += f"{edge_label}" if not no_labels else ""
                
            if use_edge_types:
                edge_types_text = ", ".join(
                    [
                        f" {edge_cls_label}: {get_edge_data(edge_data, edge_cls_label, metadata.type)} " 
                        if not no_labels else ""
                        for edge_cls_label in edge_cls_labels if edge_cls_labels
                    ]
                )
                edge_text += f"({edge_types_text})" if edge_text else edge_types_text


    return n1_text + EDGE_START + f"{edge_text}" + EDGE_END + n2_text


__all__ = ["get_edge_texts"]
