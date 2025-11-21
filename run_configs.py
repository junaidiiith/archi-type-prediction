from architype.architype.dataset.build import ArchiMateDataset, OntoUMLDataset
from architype.configs.config import RunConfig
from architype.architype.models.bert.trainer import BertTextClassifier
from tqdm.auto import tqdm

import os
import json
import itertools
import hashlib
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", type=str, default="archi", choices=["archi", "ontouml"])
    parser.add_argument("--task_type", type=str, default="node_cls", choices=["node_cls", "edge_cls", "lp"])
    parser.add_argument("--node_cls_label", type=str, default="type")
    parser.add_argument("--edge_cls_label", type=str, default="type")
    parser.add_argument("-s", type=int, default=0)
    parser.add_argument("-e", type=int, default=-1)
    return parser.parse_args()

args = parse_args()
modeling_language = args.ml

# modeling_language = 'ontouml'
dataset_name = 'eamodelset' if modeling_language == 'archi' else 'ontouml'

dataset_dir = os.path.join("architype", "data", "raw", dataset_name)
save_dir = os.path.join("results", modeling_language)

distance = [1, 0, 2, 3]
edge_removal = [0.2, 0.0, 0.9, 0.4, 0.6, 0.8]
type_semantic_removal = [0.2, 0.9, 0.4, 0.6, 0.8]
cleansed = [True, False]
ordered = [False, True]
start = args.s
end = args.e if args.e != -1 else len(cleansed)*len(ordered)*len(distance)*len(edge_removal)*len(type_semantic_removal)


for i, (distance, edge_removal, type_semantic_removal, cleansing, ordered) in tqdm(
    enumerate(itertools.product(distance, edge_removal, type_semantic_removal, cleansed, ordered)),
    total=end - start,
    desc="Configs"
):
    if i < start:
        continue
    if i >= end:
        break
    cls_label = args.node_cls_label if args.task_type == "node_cls" else args.edge_cls_label
    config_str = f"cls_label={cls_label}, distance={distance}, edge_removal={edge_removal}, type_semantic_removal={type_semantic_removal}, cleansing={cleansing}, ordered={ordered}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    config_save_dir = os.path.join(save_dir, config_hash)
    if os.path.exists(os.path.join(config_save_dir, "trainer_state.json")):
        continue
    
    os.makedirs(config_save_dir, exist_ok=True)
    config = RunConfig(
        task_type=args.task_type,
        cleanse=cleansing,
        ordered=ordered,
        distance=distance,
        edge_removal=edge_removal,
        type_semantic_removal=type_semantic_removal,
        node_cls_label=cls_label,
        edge_cls_label=cls_label,
    )
    config.save_dir = config_save_dir
    with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
        json.dump(config.model_dump(), f)
    
    
    if modeling_language == 'archi':
        dataset = ArchiMateDataset(dataset_dir, language=config.language, config=config)
    elif modeling_language == 'ontouml':
        dataset = OntoUMLDataset(dataset_dir, config=config)

    if config.edge_removal > 0 and config.edge_removal < 1:
        dataset.remove_edges(edge_removal=config.edge_removal)
        
    if config.cleanse:
        dataset.cleanse()

    if not config.ordered:
        dataset.randomize_node_labels() \
        if config.task_type == "node_cls" else \
        dataset.randomize_edge_labels()

    if config.task_type == "node_cls":
        dataset = dataset.get_node_texts(node_cls_label=cls_label)
    elif config.task_type == "edge_cls":
        dataset = dataset.get_edge_texts(edge_cls_label=cls_label)
        

    classifier = BertTextClassifier(
        model_name=config.model,
        output_dir=config.save_dir,
        seed=config.seed,
    )

    classifier.train(dataset=dataset)
    