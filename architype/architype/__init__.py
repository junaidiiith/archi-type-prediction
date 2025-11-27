"""Archi-Type Prediction package root."""

from .langgraph import LangGraph, create_graph_from_edge_index

import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

__all__ = ["LangGraph", "create_graph_from_edge_index"]

