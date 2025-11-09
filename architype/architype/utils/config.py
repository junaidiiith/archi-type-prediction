"""
GLAM4CM Configuration Settings

This module contains all configuration constants and settings used throughout
the GLAM4CM framework, including model configurations, file paths, task types,
and training parameters.

The settings are organized into logical groups:
- Model configurations (BERT, Word2Vec, FastText)
- File paths and directories
- Task type constants
- Training and evaluation constants
- Graph-specific constants

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

import os
import torch
import logging

# Configure logging for the framework
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# BERT model variants supported by the framework
BERT_MODEL = "bert-base-uncased"  # Standard BERT base model
MODERN_BERT = "answerdotai/ModernBERT-base"  # Modern BERT variant

# Traditional embedding models
WORD2VEC_MODEL = "word2vec"  # Word2Vec model identifier
TFIDF_MODEL = "tfidf"  # TF-IDF model identifier
FAST_TEXT_MODEL = "uml-fasttext.bin"  # FastText model for UML domain

# Word2Vec training configuration
W2V_CONFIG = dict(
    epoch=100,  # Number of training epochs
    dim=128,  # Embedding dimension
    ws=5,  # Window size for context
    minCount=1,  # Minimum word frequency
    thread=4,  # Number of training threads
    model="skipgram",  # Training model type (skipgram or CBOW)
)

# =============================================================================
# HARDWARE AND COMPUTATION SETTINGS
# =============================================================================

# Device configuration for PyTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("high")  # Optimize matrix multiplication precision

# Random seed for reproducibility
seed = 42

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

# Base directories
datasets_dir = "datasets"
results_dir = "results"

# Specific dataset file paths
ecore_json_path = os.path.join(datasets_dir, "ecore_555/ecore_555.jsonl")
mar_json_path = os.path.join(datasets_dir, "mar-ecore-github/ecore-github.jsonl")
modelsets_uml_json_path = os.path.join(datasets_dir, "modelset/uml.jsonl")
modelsets_ecore_json_path = os.path.join(datasets_dir, "modelset/ecore.jsonl")

# Graph data directory for processed graph representations
graph_data_dir = "datasets/graph_data"

# =============================================================================
# TASK TYPE CONSTANTS
# =============================================================================

# Downstream task identifiers
EDGE_CLS_TASK = "edge_cls"  # Edge classification task
LINK_PRED_TASK = "lp"  # Link prediction task
NODE_CLS_TASK = "node_cls"  # Node classification task
GRAPH_CLS_TASK = "graph_cls"  # Graph classification task
DUMMY_GRAPH_CLS_TASK = "dummy_graph_cls"  # Dummy task for testing

# =============================================================================
# GRAPH REPRESENTATION CONSTANTS
# =============================================================================

# Edge type constants for conceptual models
SEP = " "  # Separator for text representations
REFERENCE = "reference"  # Reference relationship type
SUPERTYPE = "supertype"  # Inheritance relationship type
CONTAINMENT = "containment"  # Composition relationship type

# =============================================================================
# TRAINING AND EVALUATION CONSTANTS
# =============================================================================

# Training phase identifiers
TRAINING_PHASE = "train"  # Training phase
VALIDATION_PHASE = "val"  # Validation phase
TESTING_PHASE = "test"  # Testing phase

# Metric names for logging and evaluation
EPOCH = "epoch"  # Epoch number
LOSS = "loss"  # Loss value
TRAIN_LOSS = "train_loss"  # Training loss
TEST_LOSS = "test_loss"  # Test loss
TEST_ACC = "test_acc"  # Test accuracy
