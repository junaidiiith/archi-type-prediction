"""
Hosted LLM API classification wrapper.
"""

from __future__ import annotations

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List
from datasets import Dataset
from pydantic import BaseModel

class ClassificationResponse(BaseModel):
    id: int
    label: str

class TextClassificationResponse(BaseModel):
    predictions: List[ClassificationResponse]


def to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def clean_label(label):
    return label.lower().replace(" ", "")

def calculate_metrics(dataset: Dataset, response: List[TextClassificationResponse]):
    y_act, y_p = list(), list()
    for actual, result in zip(dataset['label'], response):
        if not result:
            continue
        predictions = [p['label'] for p in result['predictions']]
        for p, a in zip(predictions, actual):
            if not "junction" in clean_label(a):
                y_act.append(clean_label(a))
                y_p.append(clean_label(p))
    
    def classification_metrics(y_true, y_pred):
        """
        y_true: list or array of true labels
        y_pred: list or array of predicted labels
        """
        # Unique classes
        print("Total classes: ", len(np.unique(np.concatenate([y_true, y_pred]))))
        print("Length of y_true: ", len(y_true))
        print("Length of y_pred: ", len(y_pred))
        classes = np.unique(np.concatenate([y_true, y_pred]))
        

        # --- Overall metrics ---
        overall = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),

            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),

            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # --- Per-class metrics ---
        per_class = {}
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        for i, cls in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0

            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": acc,
                "support": cm[i, :].sum()
            }

        return to_python(overall), to_python(per_class)
    
    
    return classification_metrics(y_act, y_p)