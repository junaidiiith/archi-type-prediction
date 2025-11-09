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

from ..cleanse.dedup import deduplicate_graphs
from ..cleanse.filters import (
    filter_dummy_named_graphs,
    llm_filter_graphs as llm_filter_graphs_fn,
)
from ..extract import get_edge_texts, get_node_texts
from ..langgraph.archimate import ArchiMateNxG
from ..langgraph.base import LangGraph
from ..utils.config import EDGE_CLS_TASK
from .metadata import GraphMetadata, ArchimateMetaData


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
        *,
        min_edges: int = -1,
        min_enr: float = -1,
        dummy_ratio_threshold: float = -1.0,
        duplicate_overlap_threshold: float = -1.0,
        timeout: int = -1,
        preprocess_graph_text: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.name = dataset_name
        self.metadata = metadata
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.min_edges = max(1, min_edges if min_edges > 0 else 1)
        self.min_enr = max(0.0, min_enr if min_enr > 0 else 0.0)
        self.dummy_ratio_threshold = dummy_ratio_threshold
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.timeout = timeout
        self.preprocess_graph_text = preprocess_graph_text

        self.graphs: List[LangGraph] = []

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

    @property
    def data(self):
        X, y = [], []
        for graph in self.graphs:
            X.append(graph.text)
            y.append(graph.label)

        if self.preprocess_graph_text:
            X = [self.preprocess_graph_text(x) for x in X]
        return X, y

    def get_node_texts(
        self,
        node_cls_attribute: str,
        *,
        test_size: float = 0.2,
        distance: int = 0,
        use_node_attributes: bool = False,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        use_edge_label: bool = False,
        use_node_label: bool = True,
        use_special_tokens: bool = False,
        random_state: Optional[int] = 42,
        preprocessor: Optional[Callable[[str], str]] = None,
    ) -> DatasetDict:
        """
        Build a masked node classification dataset as Hugging Face DatasetDict.
        """

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
                use_node_attributes=use_node_attributes,
                use_node_types=use_node_types,
                use_edge_types=use_edge_types,
                use_edge_label=use_edge_label,
                use_special_tokens=use_special_tokens,
                no_labels=not use_node_label,
            )
            if preprocessor is not None:
                node_text_kwargs["preprocessor"] = preprocessor

            node_texts = get_node_texts(
                nx_graph,
                distance,
                self.metadata,
                **node_text_kwargs,
            )

            for node in nodes:
                label = nx_graph.nodes[node].get(node_cls_attribute)
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
        edge_cls_attribute,
        *,
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

        rng = np.random.default_rng(random_state)
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
                label = nx_graph.edges[edge].get(edge_cls_attribute)
                if label is None:
                    continue
                edge_text_kwargs: Dict[str, Any] = dict(
                    use_node_attributes=use_node_attributes,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    use_edge_label=use_edge_label,
                    use_special_tokens=use_special_tokens,
                    no_labels=not use_node_label,
                )
                if preprocessor is not None:
                    edge_text_kwargs["preprocessor"] = preprocessor

                text = get_edge_texts(
                    nx_graph,
                    edge,
                    distance,
                    task_type=task_type,
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
        pkl_file = f"{self.name}{'_with_dummies' if self.dummy_ratio_threshold > 0 else ''}.pkl"
        with open(os.path.join(self.save_dir, pkl_file), "wb") as handle:
            pickle.dump(self.graphs, handle)
        print(f"Saved {self.name} to pickle")

    def filter_graphs(self) -> None:
        graphs: List[LangGraph] = []
        for graph in self.graphs:
            addable = True
            if self.min_edges > 0 and graph.number_of_edges() < self.min_edges:
                addable = False
            if self.min_enr > 0 and graph.enr < self.min_enr:
                addable = False
            if addable:
                graphs.append(graph)
        self.graphs = graphs

    def load(self) -> None:
        print(f"Loading {self.name} from pickle")
        pkl_file = f"{self.name}{'_with_dummies' if self.dummy_ratio_threshold > 0 else ''}.pkl"
        with open(os.path.join(self.save_dir, pkl_file), "rb") as handle:
            self.graphs = pickle.load(handle)

        self.filter_graphs()
        print(f"Loaded {self.name} with {len(self.graphs)} graphs")

    @property
    def summary(self) -> dict:
        num_graphs = len(self.graphs)
        num_edges = sum(graph.number_of_edges() for graph in self.graphs)
        num_nodes = sum(graph.number_of_nodes() for graph in self.graphs)
        average_nodes = num_nodes / num_graphs
        average_edges = num_edges / num_graphs
        average_n2e_ratio = np.mean(
            [graph.number_of_nodes() / graph.number_of_edges() for graph in self.graphs]
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
        dataset_name: str = "archimate",
        save_dir: str = ".tmp/pickles",
        reload: bool = False,
        min_edges: int = -1,
        min_enr: float = -1,
        duplicate_overlap_threshold: float = -1.0,
        dummy_ratio_threshold: float = -1.0,
        preprocess_graph_text: Optional[Callable[[str], str]] = None,
        timeout: int = -1,
        language: Optional[str] = None,
    ):
        super().__init__(
            dataset_name,
            metadata=ArchimateMetaData(),
            dataset_dir=dataset_dir,
            save_dir=save_dir,
            min_edges=min_edges,
            min_enr=min_enr,
            duplicate_overlap_threshold=duplicate_overlap_threshold,
            dummy_ratio_threshold=dummy_ratio_threshold,
            timeout=timeout,
            preprocess_graph_text=preprocess_graph_text,
        )
        os.makedirs(save_dir, exist_ok=True)
        self.duplicate_records = []
        self.flagged_dummy_graphs: List[ArchiMateNxG] = []
        self.flagged_llm_graphs: List[Tuple[ArchiMateNxG, str]] = []

        dataset_exists = os.path.exists(os.path.join(save_dir, f"{dataset_name}.pkl"))
        if reload or not dataset_exists:
            self.graphs = []
            data_path = os.path.join(dataset_dir, "processed-models")
            if language:
                df = pd.read_csv(os.path.join(dataset_dir, f"{language}-metadata.csv"))
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
            self.filter_graphs()
            self.save()
        else:
            self.load()

        if duplicate_overlap_threshold > 0:
            self.graphs, self.duplicate_records = deduplicate_graphs(
                self.graphs,
                edge_overlap_threshold=self.duplicate_overlap_threshold,
            )

        if dummy_ratio_threshold > 0:
            self.graphs, self.flagged_dummy_graphs = filter_dummy_named_graphs(
                self.graphs,
                min_edges=self.min_edges,
                min_enr=self.min_enr,
                dummy_ratio_threshold=self.dummy_ratio_threshold,
            )

        assert all(graph.number_of_edges() >= min_edges for graph in self.graphs), (
            f"Filtered out graphs with less than {min_edges} edges"
        )
        print(f"Loaded {self.name} with {len(self.graphs)} graphs")
        print(f"Graphs: {len(self.graphs)}")

    def llm_filter_graphs(
        self, threshold: float = 0.5
    ) -> Tuple[List[ArchiMateNxG], List[Tuple[ArchiMateNxG, str]]]:
        graphs, flagged = llm_filter_graphs_fn(self.graphs, threshold=threshold)
        self.flagged_llm_graphs = flagged
        return graphs, flagged

    def __repr__(self) -> str:
        return f"ArchiMateDataset({self.name}, graphs={len(self.graphs)})"


__all__ = ["ModelDataset", "ArchiMateDataset"]
