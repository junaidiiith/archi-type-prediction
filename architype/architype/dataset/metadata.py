"""
Metadata helpers describing how modeling-language artifacts map onto LangGraph
structures. These implementations are ported from the original repository and
will be expanded as the new pipeline solidifies.
"""

from __future__ import annotations


class GraphMetadata:
    def __init__(self, model_type: str):
        self.type = model_type

    @property
    def node_label(self):
        return self.node.get("label", None)

    @property
    def node_cls(self):
        return self.node.get("cls", None)

    @property
    def node_attributes(self):
        return self.node.get("attributes", None)

    @property
    def edge_label(self):
        return self.edge.get("label", None)

    @property
    def edge_cls(self):
        return self.edge.get("cls", None)

    @property
    def graph_cls(self):
        return self.graph.get("cls", None)

    @property
    def graph_label(self):
        return self.graph.get("label", None)


class EcoreMetaData(GraphMetadata):
    def __init__(self):
        super().__init__("ecore")
        self.node = {
            "label": "name",
            "cls": ["abstract"],
            "attributes": "attributes",
        }
        self.edge = {
            "label": "name",
            "cls": ["type"],
        }
        self.graph = {
            "label": "text",
            "cls": ["label"],
        }


class ArchimateMetaData(GraphMetadata):
    def __init__(self):
        super().__init__("archimate")
        self.node = {
            "label": "name",
            "cls": ["type", "layer"],
        }
        self.edge = {
            "cls": ["type"],
        }
        self.graph = {
            "label": ["text"],
        }


class OntoUMLMetaData(GraphMetadata):
    def __init__(self):
        super().__init__("ontouml")
        self.node = {
            "label": "name",
            "cls": ["stereotype"],
            "attributes": "properties",
        }
        self.edge = {
            "cls": ["type"],
        }
        self.graph = {
            "label": ["text"],
        }


__all__ = [
    "GraphMetadata",
    "EcoreMetaData",
    "ArchimateMetaData",
    "OntoUMLMetaData",
]
