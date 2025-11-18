from architype.architype.dataset.build import OntoUMLDataset
from architype.configs.config import RunConfig
from architype.architype.models.bert.trainer import BertTextClassifier

import os


config = RunConfig()
dataset_dir = os.path.join("architype", "data", "raw", "eamodelset")

archimate_dataset = OntoUMLDataset(dataset_dir, config=config)

if config.cleanse:
    archimate_dataset.cleanse()

if not config.ordered:
    archimate_dataset.randomize_node_labels() \
    if config.task_type == "node_cls" else \
    archimate_dataset.randomize_edge_labels()

if config.task_type == "node_cls":
    dataset = archimate_dataset.get_node_texts()

    classifier = BertTextClassifier(
        model_name=config.model,
        output_dir=config.save_dir,
        seed=config.seed,
    )

    classifier.train(dataset=dataset)

classifier.evaluate(dataset=dataset)