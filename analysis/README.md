# Encrypted-Traffic Intent Analysis

## Scripts
- `analysis/01_build_dataset.py`: merge shard JSON files, derive labels from `prompts/prompts.json`, build metadata-only tabular features, apply prompt-group split, run leakage audit.
- `analysis/02_train_tabular.py`: train tabular baselines (LogReg, SVM linear/RBF, RandomForest, optional XGBoost/LightGBM) with weighted training and natural-distribution evaluation.
- `analysis/03_train_lstm.py`: train LSTM over `[data_lengths, time_diffs]` sequences with deterministic truncate/pad and weighted BCE loss.
- `analysis/04_ablations.py`: run feature-family ablations (`timing`, `size`, `token`, etc.).

## Output artifacts
- Dataset table: `artifacts/datasets/openrouter_intent_features.parquet` (or `.csv` fallback)
- QC report: `artifacts/reports/dataset_qc.json`
- Tabular metrics: `artifacts/reports/tabular_metrics.json`
- LSTM metrics: `artifacts/reports/lstm_metrics.json`
- Ablation metrics: `artifacts/reports/ablation_metrics.json`
- Model files: `artifacts/models/`

## Install dependencies
Minimum:
```bash
pip install numpy pandas scikit-learn pyyaml
```

Required for full pipeline (tabular boosted models + LSTM):
```bash
pip install xgboost lightgbm torch joblib
```

Optional:
```bash
pip install pyarrow
```

Notes:
- `pyarrow` is only needed if you want parquet output; otherwise the pipeline falls back to CSV.
- Plotting libraries are not required by the current analysis scripts.

## Run order
```bash
python3 analysis/01_build_dataset.py --config analysis/config.yaml
python3 analysis/02_train_tabular.py --config analysis/config.yaml
python3 analysis/03_train_lstm.py --config analysis/config.yaml
python3 analysis/04_ablations.py --config analysis/config.yaml
```
