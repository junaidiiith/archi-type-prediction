"""Archi-Type Prediction package root."""

from .langgraph import LangGraph, create_graph_from_edge_index
import torch

# Enable TF32 (faster, slight precision tradeoff)
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"

__all__ = ["LangGraph", "create_graph_from_edge_index"]

