from collections import defaultdict
from typing import List
from datasets import DatasetDict, Dataset
import random
from .prompts import SYSTEM_PROMPT, USER_PROMPT, FEW_SHOT_USER_PROMPT


SEPARATOR = "\n\n"

def list_to_string(list_of_strings: List[str], separator: str = SEPARATOR):
    return separator.join([f"{i+1}. {text}" for i, text in enumerate(list_of_strings)])


def create_prompt_dataset(
    dataset: DatasetDict, *, 
    system_prompt: str = SYSTEM_PROMPT, 
    user_prompt: str = USER_PROMPT, 
    k: int = 0, 
    batch_size: int = 20,
    local: bool = False, 
    use_descriptions: bool = False, 
    descriptions: dict = None,
):
    if k > 0 and "{examples}" not in user_prompt:
        user_prompt = FEW_SHOT_USER_PROMPT
        
    if use_descriptions and descriptions is None:
        raise ValueError(
            "Descriptions must be provided if use_descriptions is True")

    if use_descriptions:
        system_prompt += f"\n\nThe Description of the Types are as follows:\n{descriptions}"
    

    def get_zero_shot_dataset():
        print("Getting zero shot dataset")
        messages = [{"role": "system", "content": system_prompt}]
        prompts_dataset = list()
        for i in range(0, len(dataset["test"]), batch_size):
            batch = dataset["test"][i:i+batch_size]
            batch_labels = batch['label']
            batch_texts = list_to_string(batch['text'])
            prompts_dataset.append(
                {
                    "prompt": messages + [{"role": "user", "content": user_prompt.format(text=batch_texts)}],
                    "label": batch_labels,
                }
            )
        prompts_dataset = Dataset.from_list(prompts_dataset)
        return prompts_dataset
    
    
    def get_few_shot_examples(texts, labels, k):
        assert k > 0, f"k must be positive"
        
        # import hashlib
        # texts_hash = hashlib.sha256("\n\n".join(texts).encode()).hexdigest()
        # print("Few Shots Texts hash:", texts_hash)
        
        label_to_examples = defaultdict(list)
        for text, label in zip(texts, labels):
            label_to_examples[label].append(text)
        
        few_shot_examples = dict()
        for label in sorted(few_shot_examples.keys()):
            random_examples = random.sample(label_to_examples[label], min(k, len(label_to_examples[label])))
            few_shot_examples[label] = random_examples
        
        few_shot_str = "Below are some examples of different type labels:\n"
        few_shot_str += "\n\n".join(
            [
                f"\n{'-'*int(len(label)*1.37)}\n{i+1}. {label}\n{'-'*int(1.37*len(label))}\n{list_to_string(few_shot_examples[label])}" 
                for i, label in enumerate(few_shot_examples)
            ]
        )

        # contents_hash = hashlib.sha256(few_shot_str.encode()).hexdigest()
        # print("Few shots examples hash:", contents_hash)

        return few_shot_str
    
    
    def get_few_shot_batch(test_texts: List[dict], few_shot_examples: str):
        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i:i+batch_size]
            batch_labels, batch_texts = [i['label'] for i in batch], list_to_string([i['text'] for i in batch])
            yield (
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_prompt.format(text=batch_texts, examples=few_shot_examples)}
                    ],
                    "label": batch_labels,
                }
            )    

    def get_local_few_shots_dataset():
        """
        Get a dataset of few shot examples from the training set for each graph_id in the dataset.
        """
        print("Getting local few shots dataset")
        train_texts_by_graph_id = defaultdict(list)
        for entry in dataset["train"]:
            train_texts_by_graph_id[entry["graph_id"]].append({"text": entry['text'], "label": entry['label']})
        test_texts_by_graph_id = defaultdict(list)
        
        for entry in dataset["test"]:
            test_texts_by_graph_id[entry["graph_id"]].append({"text": entry['text'], "label": entry['label']})
        
        few_shot_dataset = list()
        for graph_id in train_texts_by_graph_id:
            train_texts = train_texts_by_graph_id[graph_id]
            train_labels = [i['label'] for i in train_texts]
            train_texts = [i['text'] for i in train_texts]
            few_shot_examples = get_few_shot_examples(train_texts, train_labels, k=k)
            
            test_texts = test_texts_by_graph_id[graph_id]
            
            for batch in get_few_shot_batch(test_texts, few_shot_examples):
                few_shot_dataset.append(batch)
                
                
        few_shot_dataset = Dataset.from_list(few_shot_dataset)
        return few_shot_dataset
            
            
    def get_global_few_shots_dataset():
        """
        Get a dataset of few shot examples from the training set for each graph_id in the dataset.
        """
        print("Getting global few shots dataset")
        few_shot_examples = get_few_shot_examples(dataset["train"]["text"], dataset["train"]["label"], k=k)
        
        
        test_texts_by_graph_id = defaultdict(list)
        for entry in dataset["test"]:
            test_texts_by_graph_id[entry["graph_id"]].append({"text": entry['text'], "label": entry['label']})
        
        few_shot_dataset = list()
        for graph_id in test_texts_by_graph_id:
            test_texts = test_texts_by_graph_id[graph_id]
            
            for batch in get_few_shot_batch(test_texts, few_shot_examples):
                few_shot_dataset.append(batch)
                
        few_shot_dataset = Dataset.from_list(few_shot_dataset)
        return few_shot_dataset
    

    if k == 0:
        return get_zero_shot_dataset()
    elif local:
        return get_local_few_shots_dataset()
    else:
        return get_global_few_shots_dataset()
