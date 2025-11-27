from architype.architype.dataset.build import ArchiMateDataset, OntoUMLDataset
from architype.configs.config import RunConfig
from architype.architype.models.bert.trainer import BertTextClassifier, BertTrainingConfig
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
    parser.add_argument("--use_node_types", action="store_true")
    parser.add_argument("--use_edge_types", action="store_true")
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("-s", type=int, default=0)
    parser.add_argument("-e", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="results-ordered")
    
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    return parser.parse_args()


args = parse_args()
modeling_language = args.ml

# modeling_language = 'ontouml'
dataset_name = 'eamodelset' if modeling_language == 'archi' else 'ontouml'

dataset_dir = os.path.join("architype", "data", "raw", dataset_name)
save_dir = os.path.join(args.save_dir, modeling_language)

d2b_map = {
    'archi': {
        "node_cls": {0: 160, 1: 64, 2: 32},
        "edge_cls": {0: 160, 1: 64, 2: 32}
    },
    "ontouml": {
        "node_cls": {0: 160, 1: 64, 2: 32},
    }
}

distance = [1]
edge_removal = [0.0, 0.2]
type_semantic_removal = [0.2]
cleansed = [False]
ordered = [True, False]


start = args.s
end = args.e if args.e != -1 else len(cleansed)*len(ordered)*len(distance)*len(edge_removal)*len(type_semantic_removal)

bert_config = BertTrainingConfig(
    num_train_epochs=args.num_train_epochs,
    per_device_eval_batch_size=args.eval_batch_size,
    per_device_train_batch_size=args.train_batch_size,
)

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
    config_str = f"task_type={args.task_type}, use_node_types={args.use_node_types}, use_edge_types={args.use_edge_types}, cls_label={cls_label}, distance={distance}, edge_removal={edge_removal}, type_semantic_removal={type_semantic_removal}, cleansing={cleansing}, ordered={ordered}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    config_save_dir = os.path.join(save_dir, config_hash)
    if os.path.exists(os.path.join(config_save_dir, "trainer_state.json")):
        continue
    else:
        print("Not exists:", config_str)
    
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
        top_k=args.top_k,
    )
    config.save_dir = config_save_dir
    config.extraction_config.use_node_types = args.use_node_types
    config.extraction_config.use_edge_types = args.use_edge_types
    
    with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
        json.dump(config.model_dump(), f)
    
    
    if modeling_language == 'archi':
        dataset = ArchiMateDataset(dataset_dir, language=config.language, config=config)
    elif modeling_language == 'ontouml':
        dataset = OntoUMLDataset(dataset_dir, config=config)
    
    # dataset.filter_by_buckets(filter_by="nodes" if args.task_type == "node_cls" else "edges")

    if config.edge_removal > 0 and config.edge_removal < 1:
        dataset.remove_edges(edge_removal=config.edge_removal)
        
    if config.cleanse:
        dataset.cleanse()

    if not config.ordered:
        dataset.randomize_node_labels()

    if config.task_type == "node_cls":
        dataset = dataset.get_node_texts(node_cls_label=cls_label)
    elif config.task_type == "edge_cls":
        dataset = dataset.get_edge_texts(edge_cls_label=cls_label)
        
    bert_config.per_device_train_batch_size = d2b_map[modeling_language][config.task_type][distance]
    classifier = BertTextClassifier(
        model_name=config.model,
        output_dir=config.save_dir,
        seed=config.seed,
        config=bert_config,
    )

    classifier.train(dataset=dataset)
    