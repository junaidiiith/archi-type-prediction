from architype.architype.dataset.build import ArchiMateDataset
from architype.configs.config import RunConfig
from architype.architype.models.bert.trainer import BertTextClassifier
from tqdm.auto import tqdm

import os
import json
import itertools
import hashlib


dataset_dir = os.path.join("architype", "data", "raw", "eamodelset")


save_dir = os.path.join("results", "archi")

distance = [1, 0, 2, 3]
edge_removal = [0.2, 0.0, 0.9, 0.4, 0.6, 0.8]
type_semantic_removal = [0.2, 0.0, 0.9, 0.4, 0.6, 0.8]
cleansed = [True, False]
ordered = [False, True]


for distance, edge_removal, type_semantic_removal, cleansing, ordered in tqdm(
    itertools.product(distance, edge_removal, type_semantic_removal, cleansed, ordered),
    total=len(cleansed)*len(ordered)*len(distance)*len(edge_removal)*len(type_semantic_removal),
    desc="Configs"
):
    config = RunConfig(
        cleanse=cleansing,
        ordered=ordered,
        distance=distance,
        edge_removal=edge_removal,
        type_semantic_removal=type_semantic_removal,
    )
    config_hash = hashlib.sha256(json.dumps(config.model_dump()).encode()).hexdigest()
    config.save_dir = os.path.join(save_dir, config_hash)
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
        json.dump(config.model_dump(), f)
    
    archimate_dataset = ArchiMateDataset(dataset_dir, language=config.language, config=config)

    if config.edge_removal > 0 and config.edge_removal < 1:
        archimate_dataset.remove_edges(edge_removal=config.edge_removal)
        
    if config.cleanse:
        archimate_dataset.cleanse()

    if not config.ordered:
        archimate_dataset.randomize_node_labels() \
        if config.task_type == "node_cls" else \
        archimate_dataset.randomize_edge_labels()

    if config.task_type == "node_cls":
        dataset = archimate_dataset.get_node_texts()
    elif config.task_type == "edge_cls":
        dataset = archimate_dataset.get_edge_texts()
        

    classifier = BertTextClassifier(
        model_name=config.model,
        output_dir=config.save_dir,
        seed=config.seed,
    )

    classifier.train(dataset=dataset)