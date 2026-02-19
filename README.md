# GPTIntent

Repository layout has been organized by purpose:

- `analysis/`: ML experiments for encrypted-traffic intent inference
- `src/collection/`: dataset collection scripts
- `prompts/`: prompt-generation code and prompt datasets
- `scripts/`: operational scripts
- `env/`: environment specs
- `reproducibility/`: exported lock files and package manifests
- `data/`: collected shard data
- `artifacts/`: model/dataset/report outputs

## Quick start

### 1) Create environment

```bash
bash scripts/create_conda_env.sh
```

This installs the full analysis runtime needed by default (`xgboost`, `lightgbm`, `torch`) on top of the base conda environment.

### 2) Run analysis pipeline

```bash
python3 analysis/01_build_dataset.py --config analysis/config.yaml
python3 analysis/02_train_tabular.py --config analysis/config.yaml
python3 analysis/03_train_lstm.py --config analysis/config.yaml
python3 analysis/04_ablations.py --config analysis/config.yaml
```

### 3) Collection scripts

- Parallel setup helper: `scripts/parallel.sh`
- Fast collector: `src/collection/collect_fast.py`
- OpenRouter chatbot shim: `src/collection/gpt_5_mini_openrouter.py`

Detailed analysis instructions are in `analysis/README.md`.
