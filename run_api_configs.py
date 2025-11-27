import random
from typing import List
from architype.architype.models.api.classifier import calculate_metrics
from architype.architype.models.api.data_generator import create_prompt_dataset
from architype.architype.dataset.metadata import get_descriptions
from architype.architype.models.api.prompts import SYSTEM_PROMPT, USER_PROMPT
from architype.architype.dataset.build import ArchiMateDataset, OntoUMLDataset
from architype.configs.config import RunConfig
from tqdm.auto import tqdm
from pydantic import BaseModel
from architype.architype.llm.base import LLMService

class ClassificationResponse(BaseModel):
    id: int
    label: str

class TextClassificationResponse(BaseModel):
    predictions: List[ClassificationResponse]


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
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--use_descriptions", action="store_true")
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("-s", type=int, default=0)
    parser.add_argument("-e", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="results-api")
    
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "anthropic", "gemini", "togetherai", "deepseek"])
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    return parser.parse_args()


args = parse_args()
modeling_language = args.ml

# modeling_language = 'ontouml'
dataset_name = 'eamodelset' if modeling_language == 'archi' else 'ontouml'

dataset_dir = os.path.join("architype", "data", "raw", dataset_name)
save_dir = os.path.join(args.save_dir, modeling_language, args.llm_provider)



edge_removals = [0.0, 0.2]
type_semantic_removals = [0.2, 0.6]
cleansed_states = [False, True]
ordered_states = [True]
distances = [1, 0]


start = args.s
end = args.e if args.e != -1 else len(edge_removals)*len(type_semantic_removals)*len(cleansed_states)*len(ordered_states)*len(distances)

for i, (edge_removal, type_semantic_removal, cleansing, ordered, distance) in tqdm(
    enumerate(itertools.product(edge_removals, type_semantic_removals, cleansed_states, ordered_states, distances)),
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
    
    args.use_descriptions = True
    descriptions = get_descriptions(dataset_name, "node" if config.task_type == "node_cls" else "edge") if args.use_descriptions else None
    
    prompt_dataset = create_prompt_dataset(
        dataset,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        batch_size=args.batch_size,
        k=args.shots,
        local=args.local,
        use_descriptions=args.use_descriptions,
        descriptions=descriptions
    )
    
    if args.sample_size > 0:
        prompt_dataset = prompt_dataset.select(random.sample(range(len(prompt_dataset)), args.sample_size))
        response_str = f"bs_{args.batch_size}_k_{args.shots}_local_{int(args.local)}_use_descriptions_{int(args.use_descriptions)}"
        llm_client = LLMService(provider=args.llm_provider, model_name=args.llm_model)
        
        response = llm_client.get_llm_response_parallel(
            prompt_dataset["prompt"], 
            response_format=TextClassificationResponse,
            function_name=os.path.join(config_save_dir, response_str)
        )
        for i, r in enumerate(response):
            if r:
                with open(os.path.join(config_save_dir, f"{response_str}_{i}_actual.json"), "w") as f:
                    json.dump(prompt_dataset[i]["label"], f)
                    
        metrics = calculate_metrics(prompt_dataset, response)
        with open(os.path.join(config_save_dir, f"{response_str}_metrics.json"), "w") as f:
            json.dump(metrics, f)
        
        with open(os.path.join(config_save_dir, f"{response_str}_config.json"), "w") as f:
            json.dump(config.model_dump(), f)