"""
Dataset builders migrated from the previous ``data_loading`` package. These
classes will eventually orchestrate the full cleansing / masking pipeline.
"""

from __future__ import annotations

import json
import os
import pickle
from random import shuffle
from typing import Any, Dict, Callable, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from architype.configs.config import RunConfig


from ..cleanse.dedup import deduplicate_graphs, filter_by_edges
from ..cleanse.filters import (
    filter_dummy_named_graphs,
    llm_filter_graphs as llm_filter_graphs_fn,
)
from ..extract import get_edge_texts, get_node_texts
from ..langgraph.archimate import ArchiMateNxG
from ..langgraph.ontouml import OntoUMLNxG
from ..langgraph.base import LangGraph
from ..utils.config import EDGE_CLS_TASK
from .metadata import GraphMetadata, ArchimateMetaData, OntoUMLMetaData


def _to_dataset(samples: List[Dict[str, Any]]) -> Dataset:
    if not samples:
        return Dataset.from_dict({"text": [], "label": []})
    columns: Dict[str, List[Any]] = {}
    keys = samples[0].keys()
    for key in keys:
        columns[key] = [sample.get(key) for sample in samples]
    return Dataset.from_dict(columns)


class ModelDataset:
    def __init__(
        self,
        dataset_name: str,
        metadata: GraphMetadata,
        dataset_dir: str = ".tmp/datasets",
        save_dir: str = ".tmp/pickles",
        config: Optional[RunConfig] = None,
        timeout: int = -1,
    ) -> None:
        self.name = dataset_name
        self.metadata = metadata
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.timeout = timeout
        self.graphs: List[LangGraph] = []
        
        self.config = config

    def get_train_test_split(
        self, train_size: float = 0.8
    ) -> Tuple[List[int], List[int]]:
        n = len(self.graphs)
        train_size = int(n * train_size)
        idx = list(range(n))
        shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        return train_idx, test_idx

    def k_fold_split(self, k: int = 10, seed: int = 42):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        n = len(self.graphs)
        for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):
            yield train_idx, test_idx

    def get_data(self, preprocessor: Optional[Callable[[str], str]] = None):
        data = []
        for graph in self.graphs:
            data.append(graph.text)

        if preprocessor is not None:
            data = [preprocessor(x) for x in data]

        return Dataset.from_list(data)

    def get_node_texts(
        self,
        *,
        node_cls_label: str = "type",
        test_size: float = 0.2,
        distance: int = 0,
        use_node_attributes: bool = True,
        use_node_types: bool = True,
        use_edge_types: bool = True,
        use_edge_label: bool = True,
        use_node_label: bool = True,
        use_special_tokens: bool = False,
        random_state: Optional[int] = 42,
        preprocessor: Optional[Callable[[str], str]] = None,
    ) -> DatasetDict:
        """
        Build a masked node classification dataset as Hugging Face DatasetDict.
        """

        node_cls_label = self.config.node_cls_label if self.config else node_cls_label
        rng = np.random.default_rng(random_state)
        train_samples: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []

        for nx_graph in tqdm(self.graphs, desc="Getting node texts"):
            nodes = list(nx_graph.nodes)
            if not nodes:
                continue

            if test_size <= 0:
                num_test = 0
            elif test_size >= 1:
                num_test = len(nodes)
            else:
                num_test = int(np.floor(len(nodes) * test_size))
                if num_test == 0:
                    num_test = 1
            num_test = min(num_test, len(nodes))

            if num_test > 0:
                indices = rng.choice(len(nodes), size=num_test, replace=False)
                test_nodes = {nodes[i] for i in indices}
            else:
                test_nodes = set()

            for node in nodes:
                nx_graph.nodes[node]["masked"] = node in test_nodes

            node_text_kwargs: Dict[str, Any] = dict(
                use_node_attributes=self.config.extraction_config.use_node_attributes if self.config else use_node_attributes,
                use_node_types=self.config.extraction_config.use_node_types if self.config else use_node_types,
                use_edge_types=self.config.extraction_config.use_edge_types if self.config else use_edge_types,
                use_edge_label=self.config.extraction_config.use_edge_label if self.config else use_edge_label,
                use_special_tokens=self.config.extraction_config.use_special_tokens if self.config else use_special_tokens,
                no_labels=not self.config.extraction_config.use_node_label if self.config else not use_node_label,
            )
            if preprocessor is not None:
                node_text_kwargs["preprocessor"] = preprocessor

            node_texts = get_node_texts(
                nx_graph,
                d=self.config.distance if self.config else distance,
                metadata=self.metadata,
                **node_text_kwargs,
            )

            for node in nodes:
                label = nx_graph.nodes[node].get(node_cls_label)
                if label is None:
                    continue
                sample = {
                    "text": node_texts.get(node, ""),
                    "label": label,
                    "graph_id": getattr(
                        nx_graph, "graph_id", getattr(nx_graph, "id", None)
                    ),
                    "node_id": node,
                }
                if node in test_nodes:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

        return DatasetDict(
            {
                "train": _to_dataset(train_samples),
                "test": _to_dataset(test_samples),
            }
        )

    def get_edge_texts(
        self,
        *,
        edge_cls_label: str = "type",
        test_size: float = 0.2,
        distance: int = 0,
        use_node_attributes: bool = False,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        use_edge_label: bool = True,
        use_node_label: bool = True,
        use_special_tokens: bool = False,
        task_type: str = EDGE_CLS_TASK,
        random_state: Optional[int] = None,
        preprocessor: Optional[Callable[[str], str]] = None,
    ) -> DatasetDict:
        """
        Build a masked edge classification dataset as Hugging Face DatasetDict.
        """

        edge_cls_label = self.config.edge_cls_label if self.config else edge_cls_label
        rng = np.random.default_rng(self.config.seed if self.config else random_state)
        train_samples: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []

        for nx_graph in tqdm(self.graphs, desc="Getting edge texts"):
            edges = list(nx_graph.edges)
            if not edges:
                continue

            if test_size <= 0:
                num_test = 0
            elif test_size >= 1:
                num_test = len(edges)
            else:
                num_test = int(np.floor(len(edges) * test_size))
                if num_test == 0:
                    num_test = 1
            num_test = min(num_test, len(edges))

            if num_test > 0:
                indices = rng.choice(len(edges), size=num_test, replace=False)
                test_edges = {edges[i] for i in indices}
            else:
                test_edges = set()

            for edge in edges:
                nx_graph.edges[edge]["masked"] = edge in test_edges

            for edge in edges:
                label = nx_graph.edges[edge].get(edge_cls_label)
                if label is None:
                    continue
                edge_text_kwargs: Dict[str, Any] = dict(
                    use_node_attributes=self.config.extraction_config.use_node_attributes if self.config else use_node_attributes,
                    use_node_types=self.config.extraction_config.use_node_types if self.config else use_node_types,
                    use_edge_types=self.config.extraction_config.use_edge_types if self.config else use_edge_types,
                    use_edge_label=self.config.extraction_config.use_edge_label if self.config else use_edge_label,
                    use_special_tokens=self.config.extraction_config.use_special_tokens if self.config else use_special_tokens,
                    no_labels=not self.config.extraction_config.use_node_label if self.config else not use_node_label,
                )
                if preprocessor is not None:
                    edge_text_kwargs["preprocessor"] = preprocessor

                text = get_edge_texts(
                    nx_graph,
                    edge,
                    d=self.config.distance if self.config else distance,
                    task_type=self.config.task_type if self.config else task_type,
                    metadata=self.metadata,
                    **edge_text_kwargs,
                )

                sample = {
                    "text": text,
                    "label": label,
                    "graph_id": getattr(
                        nx_graph, "graph_id", getattr(nx_graph, "id", None)
                    ),
                    "edge": edge,
                }
                if edge in test_edges:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

        return DatasetDict(
            {
                "train": _to_dataset(train_samples),
                "test": _to_dataset(test_samples),
            }
        )

    def randomize_node_labels(
        self, *, random_state: Optional[int] = None
    ) -> List[LangGraph]:
        """
        Create deep copies of the dataset graphs and shuffle the metadata-defined
        node label attribute across nodes within each graph.
        """
        node_label_attr = getattr(self.metadata, "node_label", None)
        if not node_label_attr:
            raise ValueError(
                "Metadata does not define a node label attribute.")

        rng = np.random.default_rng(random_state)
        randomized_graphs: List[LangGraph] = []

        for graph in self.graphs:
            cloned_graph = graph.copy()
            labeled_nodes = [
                node for node in cloned_graph.nodes
                if node_label_attr in cloned_graph.nodes[node]
            ]
            labels = [cloned_graph.nodes[node][node_label_attr]
                      for node in labeled_nodes]
            if not labels:
                randomized_graphs.append(cloned_graph)
                continue

            rng.shuffle(labels)
            for node, label in zip(labeled_nodes, labels):
                cloned_graph.nodes[node][node_label_attr] = label
            randomized_graphs.append(cloned_graph)

        return randomized_graphs

    def randomize_edge_labels(
        self, *, random_state: Optional[int] = None
    ) -> List[LangGraph]:
        """
        Create deep copies of the dataset graphs and shuffle the metadata-defined
        edge label attribute across edges within each graph.
        """
        edge_label_attr = getattr(self.metadata, "edge_label", None)
        if not edge_label_attr:
            raise ValueError(
                "Metadata does not define an edge label attribute.")

        rng = np.random.default_rng(random_state)
        randomized_graphs: List[LangGraph] = []

        for graph in self.graphs:
            cloned_graph = graph.copy()
            labeled_edges = [
                edge for edge in cloned_graph.edges
                if edge_label_attr in cloned_graph.edges[edge]
            ]
            labels = [cloned_graph.edges[edge][edge_label_attr]
                      for edge in labeled_edges]
            if not labels:
                randomized_graphs.append(cloned_graph)
                continue

            rng.shuffle(labels)
            for edge, label in zip(labeled_edges, labels):
                cloned_graph.edges[edge][edge_label_attr] = label
            randomized_graphs.append(cloned_graph)

        return randomized_graphs

    def __repr__(self) -> str:
        return f"Dataset({self.name}, graphs={len(self.graphs)})"

    def __getitem__(self, key) -> LangGraph:
        return self.graphs[key]

    def __iter__(self):
        return iter(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def save(self) -> None:
        print(f"Saving {self.name} to pickle")
        pkl_file = f"{self.name}.pkl"
        with open(os.path.join(self.save_dir, pkl_file), "wb") as handle:
            pickle.dump(self.graphs, handle)
        print(f"Saved {self.name} to pickle")

    def load(self) -> None:
        print(f"Loading {self.name} from pickle")
        pkl_file = f"{self.name}.pkl"
        with open(os.path.join(self.save_dir, pkl_file), "rb") as handle:
            self.graphs = pickle.load(handle)

        print(f"Loaded {self.name} with {len(self.graphs)} graphs")

    def cleanse(
        self,
        *,
        duplicate_overlap_threshold: float = -1,
        dummy_ratio_threshold: float = -1,
        min_edges: int = -1,
        min_enr: float = -1,
        llm_filter_threshold: float = -1,
    ) -> None:
        min_edges = self.config.cleansing_config.min_edges if self.config else min_edges
        min_enr = self.config.cleansing_config.min_enr if self.config else min_enr
        duplicate_overlap_threshold = self.config.cleansing_config.duplicate_overlap_threshold if self.config else duplicate_overlap_threshold
        dummy_ratio_threshold = self.config.cleansing_config.dummy_ratio_threshold if self.config else dummy_ratio_threshold
        llm_filter_threshold = self.config.cleansing_config.llm_filter_threshold if self.config else llm_filter_threshold
        
        self.graphs = filter_by_edges(
            self.graphs,
            min_edges=min_edges,
            min_enr=min_enr,
        )
        if duplicate_overlap_threshold > 0:
            self.graphs, _ = deduplicate_graphs(
                self.graphs,
                edge_overlap_threshold=duplicate_overlap_threshold,
            )

        if dummy_ratio_threshold > 0:
            self.graphs, _ = filter_dummy_named_graphs(
                self.graphs,
                min_edges=min_edges,
                min_enr=min_enr,
                dummy_ratio_threshold=dummy_ratio_threshold,
            )

        assert all(graph.number_of_edges() >= min_edges for graph in self.graphs), (
            f"Filtered out graphs with less than {min_edges} edges"
        )
        if llm_filter_threshold > 0:
            self.graphs, _ = llm_filter_graphs_fn(
                self.graphs,
                threshold=llm_filter_threshold,
            )
        print(f"Cleansed {self.name} with {len(self.graphs)} graphs")
        print(self.summary)

    def llm_filter_graphs(
        self, threshold: float = 0.5
    ) -> Tuple[List[LangGraph], List[Tuple[LangGraph, str]]]:
        self.graphs, flagged = llm_filter_graphs_fn(
            self.graphs, threshold=threshold)
        return flagged

    @property
    def summary(self) -> dict:
        num_graphs = len(self.graphs)
        num_edges = sum(graph.number_of_edges() for graph in self.graphs)
        num_nodes = sum(graph.number_of_nodes() for graph in self.graphs)
        average_nodes = num_nodes / num_graphs
        average_edges = num_edges / num_graphs
        average_n2e_ratio = np.mean(
            [graph.number_of_nodes() / graph.number_of_edges()
             for graph in self.graphs]
        )
        return {
            "num_graphs": num_graphs,
            "num_edges": num_edges,
            "num_nodes": num_nodes,
            "average_nodes": f"{average_nodes:.2f}",
            "average_edges": f"{average_edges:.2f}",
            "average_n2e_ratio": f"{average_n2e_ratio:.2f}",
        }


class ArchiMateDataset(ModelDataset):
    def __init__(
        self,
        dataset_dir: str,
        *,
        config: Optional[RunConfig] = None,
        dataset_name: str = "archimate",
        save_dir: str = ".tmp/pickles",
        reload: bool = False,
        timeout: int = -1,
        language: Optional[str] = None,
    ):
        super().__init__(
            dataset_name,
            metadata=ArchimateMetaData(),
            dataset_dir=dataset_dir,
            save_dir=save_dir,
            timeout=timeout,
            config=config,
        )
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(
            os.path.join(save_dir, f"{dataset_name}.pkl"))
        if reload or not dataset_exists:
            self.graphs = []
            data_path = os.path.join(dataset_dir, "processed-models")
            if language:
                df = pd.read_csv(os.path.join(
                    dataset_dir, f"{language}-metadata.csv"))
                model_dirs = df["ID"].to_list()
            else:
                model_dirs = os.listdir(data_path)

            for model_dir in tqdm(model_dirs, desc=f"Loading {dataset_name.title()}"):
                model_dir = os.path.join(data_path, model_dir)
                if not os.path.isdir(model_dir):
                    continue
                model_file = os.path.join(model_dir, "model.json")
                if not os.path.exists(model_file):
                    continue
                model = json.load(open(model_file))
                nxg = ArchiMateNxG(
                    model,
                    path=model_file,
                    timeout=timeout,
                )
                if nxg.number_of_edges() < 1:
                    continue
                self.graphs.append(nxg)

            print("Total graphs:", len(self.graphs))
            self.save()
        else:
            self.load()

    def __repr__(self) -> str:
        return f"ArchiMateDataset({self.name}, graphs={len(self.graphs)})"


class OntoUMLDataset(ModelDataset):
    def __init__(
        self,
        dataset_dir: str,
        *,
        config: Optional[RunConfig] = None,
        dataset_name: str = "ontouml",
        save_dir: str = ".tmp/pickles",
        reload: bool = False,
        timeout: int = -1,
    ):
        super().__init__(
            dataset_name,
            metadata=OntoUMLMetaData(),
            dataset_dir=dataset_dir,
            save_dir=save_dir,
            timeout=timeout,
            config=config,
        )
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(
            os.path.join(save_dir, f"{dataset_name}.pkl")
        )
        if reload or not dataset_exists:
            self.graphs = []
            data_path = os.path.join(dataset_dir, "models")
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"OntoUML data directory not found: {data_path}"
                )
            model_dirs = os.listdir(data_path)

            for model_dir in tqdm(model_dirs, desc=f"Loading {dataset_name.title()}"):
                model_dir = os.path.join(data_path, model_dir)
                if not os.path.isdir(model_dir):
                    continue
                model_file = os.path.join(model_dir, "ontology.json")
                if not os.path.exists(model_file):
                    continue
                with open(model_file, encoding="iso-8859-1") as f:
                    model = json.load(f)
                try:
                    nxg = OntoUMLNxG(model)
                    if nxg.number_of_edges() < 1:
                        continue
                    self.graphs.append(nxg)
                except Exception as exc:
                    print(f"Error in {model_file} {exc}")

            self.save()
        else:
            self.load()

        print(f"Loaded {self.name} with {len(self.graphs)} graphs")
        print(f"Graphs: {len(self.graphs)}")

    def __repr__(self) -> str:
        return f"OntoUMLDataset({self.name}, graphs={len(self.graphs)})"


__all__ = ["ModelDataset", "ArchiMateDataset", "OntoUMLDataset"]
