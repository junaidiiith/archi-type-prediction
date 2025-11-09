## Archi-Type Prediction

This repository operationalizes the study of whether large language models rely on natural-language semantics or structural regularities to infer ArchiMate element and relationship types. It packages data curation, graph transformation, textual extraction, and multiple modeling backends (BERT, Unsloth-fine-tuned LLMs, hosted APIs) into a reproducible pipeline for masked node classification on the EAModelSet corpus.

### Repository Layout

- `configs/`: YAML specifications for data pipelines, extraction toggles, training hyperparameters, evaluation metrics, and hosted LLM settings.
- `data/`: Data lake organized as `raw/`, `interim/`, and `processed/` to track EAModelSet transformations.
- `architype/langgraph/`: LangGraph abstractions (base class, ArchiMate adapter) backed by NetworkX.
- `architype/cleanse/`: Duplicate removal and filtering stubs that will enforce dataset quality thresholds.
- `architype/extract/`: Node and edge text extraction modules implementing Eq. (1)–(2) with semantic control switches.
- `architype/dataset/`: Builders (`build.py`) and mask scaffolding (`mask.py`) for constructing training/evaluation splits.
- `architype/models/`: Backends for BERT fine-tuning, Unsloth QLoRA, and hosted API inference (logic to be filled in upcoming steps).
- `architype/eval/`: Metrics, reporting, and statistical testing placeholders ready for evaluation logic.
- `architype/utils/`: Shared helpers (hashing, traversal, config constants) plus stubs for IO/logging/seed management.
- `runs/`: Logs, artifacts, and reports emitted by experiments.
- `docker/`, `Makefile`, `pyproject.toml`: Environment provisioning and packaging scaffolding.

### End-to-End Pipeline

- **Ingest**: Load EAModelSet archives, unify identifiers, and attach ArchiMate metadata from `en-metadata.csv`.
- **Transform**: Convert each source model to a LangGraph (`construct_graph`) with numeric node/edge IDs and type labels in `cls`.
- **Filter**: Enforce graph-level thresholds (`min_edges`, `min_enr`) and optional duplicate removal before split generation.
- **Extract**: Produce textual views `T_n` and `T_e` per Eq. (1)–(2) under multiple semantics controls (labels, attributes, types).
- **Mask & Split**: Create stratified train/validation/test splits while hiding configurable percentages of node `cls` labels.
- **Train**: Run chosen backend (BERT, Unsloth LLM, hosted API) with modality-specific prompts/heads.
- **Evaluate**: Aggregate metrics, ablations (semantic noise vs placeholder labels), and cost/resource snapshots into reports.

### Dataset Preparation

- `python -m architype.dataset.build --input data/raw/eamodelset --output data/interim/graphs` (roadmap) builds LangGraph pickles plus JSON manifests.
- `--clean-labels` toggles label standardization; `--use-placeholder-labels` swaps in obfuscated tokens to isolate structural learning.
- `--remove-duplicates` removes graph isomorphs via hash fingerprints; `--min-edges` and `--min-enr` gate sparse or degenerate models.
- Output artifacts include `graphs.jsonl` (metadata), `nodes.parquet` (per-node text and cls), and `splits.json` (masking plans).

### Node & Edge Text Extraction

- Configure extraction parameters in `configs/extract.yaml` specifying `k`, node signals (label, attributes, type), edge signals (label, type), and casing/normalization options.
- `python -m architype.extract.build_text_dataset --graph data/interim/graphs --profile configs/extract.yaml` (upcoming) produces text corpora for downstream training.
- Text representations follow  
  `T_n = σ(n) ∪ ⋃_{d=1}^{k} ⋃_{m∈N_d(n)} σ(m)`  
  enabling controlled contextual breadth to test semantics vs topology.

### Modeling Tracks

- **BERT Fine-Tuning (`architype/models/bert/`)**
  - Uses ModernBERT with LoRA adapters for efficiency; no prompt formatting required.
  - Planned launch: `python -m architype.models.bert.train --dataset data/processed/persona_k1.parquet --mask-rate 0.3`.
  - Evaluation logs will land in `runs/logs/bert/<timestamp>`.
- **Unsloth LLM Fine-Tuning (`architype/models/unsloth/`)**
  - Builds conversation-style prompts (optionally with thinking tokens) and applies QLoRA through Unsloth’s `FastLanguageModel`.
  - Profiles (`configs/train_unsloth.yaml`) declare base model (e.g., `unsloth/Qwen3-4B-Thinking-2507`), instruction templates, and response masking.
  - Planned launch: `python -m architype.models.unsloth.train --config configs/train_unsloth.yaml`.
  - Reference scripts in `unsloth-finetuning/` document prompt templates and last-token loss tricks.
- **Hosted LLM Prompting (`architype/models/api/`)**
  - Supports GPT-5, Claude 3.5, and other OpenAI/Anthropic APIs with zero-shot and few-shot templates stored under `architype/models/unsloth/prompts/`.
  - Planned launch: `python -m architype.models.api.run_gpt --config configs/api_llms.yaml`.

### Experimental Matrix

- **Semantic Signal Ablation**: Raw labels, cleaned labels, placeholder labels.
- **Mask Ratios**: Evaluate performance as 10%–90% of nodes remain labeled to study completion capability.
- **Hop Radius (`k`)**: Compare k ∈ {0,1,2} to gauge contextual dependence.
- **Model Scale**: BERT (hundreds of millions), mid-scale QLoRA models (3–7B), hosted frontier models.
- **Cost Tracking**: Capture GPU hours, VRAM usage, and API spend per experiment for practical deployment analysis.

### Evaluation & Reporting

- `python -m architype.eval.report --runs runs --output runs/reports/summary.csv` (planned) merges metrics across factors.
- Generates confusion matrices, calibration plots, and statistical tests (paired bootstraps) to assess significance between semantic settings.
- `reports/` also stores qualitative error analyses mapping misclassified nodes back to LangGraph neighborhoods for manual inspection.

### Environment Setup

- Python ≥3.10, CUDA-compatible GPU for local fine-tuning (tested on 24GB).
- Install base dependencies: `pip install -r requirements.txt`.
- For Unsloth tracks ensure `pip install unsloth bitsandbytes trl>=0.22`.
- API experiments require provider-specific environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Reproducing Key Results

- Run `make data` to download/process EAModelSet (requires acceptance of license).
- Run `make experiments MODEL=bert` or `make experiments MODEL=unsloth` for automated sweeps defined in `Makefile` recipes.
- Use `make report` to compile tables/figures mirroring the paper’s sections (semantic ablation, completeness, scale trade-offs).

### Contributing

- Open issues for new modeling languages by extending `LangGraph.construct_graph`.
- Submit PRs with unit tests under `tests/` and add your run summaries to `reports/notes/`.
- Join discussions on improving prompt templates or extending to link prediction tasks.

### References

- Djelic et al., “Automated Data Cleansing Pipeline for Enterprise Architecture Models,” 2025.
- SciPy Proceedings 2011: NetworkX Documentation.
- Unsloth AI Documentation for QLoRA-based fine-tuning.

---

Maintained by the Archi-Type Prediction team. Contributions, feedback, and replication reports are welcome.
