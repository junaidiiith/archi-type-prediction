"""
Cases - 
x%Structure + y%Semantics

x = % of edges removed
y = % of type semantics removed

semantics = (cleansed, ordered)
"""

from pydantic import BaseModel, Field

class CleansingConfig(BaseModel):
    min_edges: int = Field(default=5)
    min_enr: float = Field(default=-1)
    duplicate_overlap_threshold: float = Field(default=0.9)
    dummy_ratio_threshold: float = Field(default=0.5)
    llm_filter_threshold: float = Field(default=-1)


class ExtractionConfig(BaseModel):
    use_node_attributes: bool = Field(default=True)
    use_node_types: bool = Field(default=True)
    use_edge_types: bool = Field(default=True)
    use_edge_label: bool = Field(default=True)
    use_node_label: bool = Field(default=True)
    use_special_tokens: bool = Field(default=False)


class RunConfig(BaseModel):
    task_type: str = Field(default="node_cls")
    
    node_cls_label: str = Field(default="type")
    edge_cls_label: str = Field(default="type")
    
    edge_removal: float = Field(default=0.5, ge=0.0, le=1.0)
    type_semantic_removal: float = Field(default=0.5, ge=0.0, le=1.0)
    distance: int = Field(default=1, ge=0, le=3)
    cleanse: bool = Field(default=False)
    ordered: bool = Field(default=False)
    language: str = Field(default="en")
    cleansing_config: CleansingConfig = Field(default=CleansingConfig())
    extraction_config: ExtractionConfig = Field(default=ExtractionConfig())
    llm_cleansing: bool = Field(default=False)
    
    topk: int = Field(default=-1)
    
    model: str = Field(default="bert-base-uncased")
    max_seq_length: int = Field(default=4096)
    
    learning_rate: float = Field(default=2e-4)
    weight_decay: float = Field(default=0.001)
    lr_scheduler_type: str = Field(default="linear")
    warmup_steps: int = Field(default=5)
    max_steps: int = Field(default=60)
    gradient_accumulation_steps: int = Field(default=4)
    
    seed: int = Field(default=3407)
    save_dir: str = Field(default="results")
