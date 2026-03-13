# Encrypted-Traffic Intent Analysis

## Scripts
- `analysis/01_build_dataset.py`: merge shard JSON files, derive labels from `prompts/prompts.json`, build metadata-only tabular features, apply prompt-group split, run leakage audit.
- `analysis/02_train_tabular.py`: train tabular baselines (LogReg, SVM linear/RBF, RandomForest, optional XGBoost/LightGBM) with weighted training and natural-distribution evaluation.
- `analysis/03_train_lstm.py`: train LSTM over `[data_lengths, time_diffs]` sequences with deterministic truncate/pad and weighted BCE loss. Supports optional improvements: packed sequences, log/normalized inputs, bidirectional encoder, pooling modes, scheduler, gradient clipping, and early stopping.
- `analysis/04_ablations.py`: run feature-family ablations (`timing`, `size`, `token`, etc.).
- `analysis/05_lstm_method_sweep.py`: run a preset sweep of LSTM improvement strategies and rank methods by AUPRC/F1.
- `analysis/06_group_cv_tabular.py`: run group-stratified cross-validation for tabular models (grouped by prompt ID).
- `analysis/07_error_analysis.py`: produce qualitative error summaries (FP/FN profiles and top confident mistakes) from saved predictions.
- `analysis/08_make_report_figures.py`: generate report-ready plots (AUPRC bar chart, PR/ROC curves, confusion matrices) from artifacts.
- `analysis/09_early_observation_attack.py`: evaluate a frozen XGBoost model on prefix-only packet observations (5\%..100\%) and write early-leakage metrics.
- `analysis/10_blend_models.py`: learn validation-selected blending weights over saved model predictions and evaluate on test.
- `analysis/11_run_all_models.py`: one-command runner for dataset build, tabular baselines, LSTM, blending, and consolidated summary output.
- `analysis/12_cross_provider_transfer.py`: train on one provider and evaluate transfer to the other (both directions), including CI estimates.
- `analysis/13_strict_leakage_ablations.py`: run stricter leakage-control ablations (no trial/count/length/duration proxies) and quantify residual timing signal.
- `analysis/14_provider_uncertainty_and_plot.py`: compute CI summaries for key models from saved predictions and generate unpatched-vs-patched comparison plot.

## Output artifacts
- Dataset table: `artifacts/datasets/openrouter_intent_features.parquet` (or `.csv` fallback)
- QC report: `artifacts/reports/dataset_qc.json`
- Tabular metrics: `artifacts/reports/tabular_metrics.json`
- LSTM metrics: `artifacts/reports/lstm_metrics.json`
- LSTM sweep ranking: `artifacts/reports/lstm_method_sweep.json`
- Ablation metrics: `artifacts/reports/ablation_metrics.json`
- Early-observation metrics: `artifacts/reports/early_observation_metrics.json` and `.csv`
- Cross-provider transfer metrics: `artifacts/reports/cross_provider_transfer.json`
- Strict leakage-control ablations: `artifacts/reports/strict_leakage_ablations.json`
- Provider uncertainty summary + drop figure: `artifacts/reports/provider_uncertainty_summary.json`, `artifacts/reports/figures/provider_drop_side_by_side.png`
- Model files: `artifacts/models/`

## Install dependencies
Minimum:
```bash
pip install "numpy<2" pandas scikit-learn pyyaml
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
- If using `torch==2.2.x`, pin NumPy to `<2` to avoid ABI issues (`pip install "numpy<2"`).

## Run order
```bash
python3 analysis/01_build_dataset.py --config analysis/config.yaml
python3 analysis/02_train_tabular.py --config analysis/config.yaml
python3 analysis/03_train_lstm.py --config analysis/config.yaml
python3 analysis/04_ablations.py --config analysis/config.yaml
python3 analysis/06_group_cv_tabular.py --config analysis/config.yaml --folds 5 --models logreg
python3 analysis/02_train_tabular.py --config analysis/config.yaml --predictions-dir artifacts/reports/predictions
python3 analysis/07_error_analysis.py --predictions artifacts/reports/predictions/xgboost_predictions.json --model xgboost
python3 analysis/09_early_observation_attack.py --config analysis/config.yaml
python3 analysis/10_blend_models.py --pred-dir artifacts/reports/predictions --models xgboost,lightgbm,random_forest
python3 analysis/08_make_report_figures.py
```

## One-command full eval
```bash
# Run all models (build dataset -> tabular baselines -> LSTM -> blend -> summary)
python3 analysis/11_run_all_models.py \
  --config analysis/config.yaml \
  --data-glob "data/shard*/GPT5MiniOpenRouter_shard*of50.json" \
  --run-name full_eval_default

# Faster variant for LSTM (keeps tabular full)
python3 analysis/11_run_all_models.py \
  --config analysis/config.yaml \
  --data-glob "data/shard*/GPT5MiniOpenRouter_shard*of50.json" \
  --lstm-profile fast \
  --tabular-models logreg,random_forest,xgboost,lightgbm \
  --run-name full_eval_fast_lstm
```

## LSTM exploration examples
```bash
# Baseline-equivalent run
python3 analysis/03_train_lstm.py --config analysis/config.yaml

# Stronger LSTM setup (good first upgrade)
python3 analysis/03_train_lstm.py \
  --config analysis/config.yaml \
  --epochs 12 \
  --hidden-size 96 \
  --num-layers 2 \
  --bidirectional \
  --pooling attention \
  --dropout 0.3 \
  --log1p-inputs \
  --normalize-inputs \
  --scheduler plateau \
  --early-stop-patience 2 \
  --min-epochs 4 \
  --grad-clip 1.0 \
  --weight-decay 1e-4

# Run preset method sweep (fast profile by default)
python3 analysis/05_lstm_method_sweep.py --config analysis/config.yaml
# (fast profile uses shorter sequences and capped split sizes for faster turnaround)

# Full exhaustive sweep (slower)
python3 analysis/05_lstm_method_sweep.py --config analysis/config.yaml --profile full

# Tiny sanity sweep with hard time budget
python3 analysis/05_lstm_method_sweep.py --config analysis/config.yaml --profile tiny --timeout-minutes 20
```
