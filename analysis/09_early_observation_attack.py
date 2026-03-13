#!/usr/bin/env python3
"""Evaluate a frozen XGBoost model under early-observation packet prefixes."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FRACTIONS: List[float] = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]


def _require_dependencies() -> None:
    try:
        import joblib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "joblib, numpy, pandas, and scikit-learn are required. "
            "Install with: pip install joblib numpy pandas scikit-learn"
        ) from ex


def _evaluate(y_true, y_score, threshold: float, precision_ks) -> Dict[str, object]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from analysis.common import precision_at_k

    y_pred = (np.array(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "precision_at_k": {
            f"p@{int(k * 100)}": float(precision_at_k(y_true, y_score, k)) for k in precision_ks
        },
    }


def _resolve_dataset_path(config, explicit_path: str) -> str:
    if explicit_path:
        return explicit_path
    candidates = sorted(glob.glob(f"{config.paths.dataset_dir}/openrouter_intent_features.*"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find artifacts/datasets/openrouter_intent_features.*; run 01_build_dataset.py first"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run early-observation attack analysis with frozen XGBoost")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument("--dataset", default="", help="Optional explicit dataset path.")
    parser.add_argument("--tabular-metrics", default="artifacts/reports/tabular_metrics.json")
    parser.add_argument("--model-path", default="artifacts/models/xgboost.joblib")
    parser.add_argument("--output-json", default="artifacts/reports/early_observation_metrics.json")
    parser.add_argument("--output-csv", default="artifacts/reports/early_observation_metrics.csv")
    args = parser.parse_args()

    _require_dependencies()

    import joblib
    import numpy as np
    import pandas as pd

    from analysis.common import (
        DatasetRow,
        find_threshold_for_best_f1,
        load_config,
        load_labeled_rows,
        load_tabular_table,
        print_env_versions,
        row_to_feature_dict,
        save_json,
        set_seed,
    )

    config = load_config(args.config)
    set_seed(config.training.random_seed)

    dataset_path = _resolve_dataset_path(config, args.dataset)
    dataset_df = load_tabular_table(dataset_path)
    required_cols = {"sample_id", "split", "label"}
    missing_cols = sorted(required_cols - set(dataset_df.columns))
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing_cols)}")

    split_by_sample = dict(zip(dataset_df["sample_id"].tolist(), dataset_df["split"].tolist()))
    label_by_sample = dict(zip(dataset_df["sample_id"].tolist(), dataset_df["label"].astype(int).tolist()))

    with open(args.tabular_metrics, "r", encoding="utf-8") as f:
        tabular_metrics = json.load(f)
    feature_cols = list(tabular_metrics["feature_columns"])
    if "xgboost" not in tabular_metrics["models"]:
        raise ValueError("xgboost metrics not present in tabular metrics payload.")

    model = joblib.load(args.model_path)

    rows, load_stats = load_labeled_rows(config)
    val_features = {f: [] for f in FRACTIONS}
    test_features = {f: [] for f in FRACTIONS}
    y_val: List[int] = []
    y_test: List[int] = []

    kept = 0
    for row in rows:
        split = split_by_sample.get(row.sample_id, "")
        if split not in {"val", "test"}:
            continue

        # Trust split labels from the tabular dataset for consistency.
        label = int(label_by_sample[row.sample_id])
        if label != int(row.label):
            continue

        kept += 1
        for frac in FRACTIONS:
            n = max(1, int(math.ceil(len(row.data_lengths) * frac)))
            prefix_row = DatasetRow(
                sample_id=row.sample_id,
                prompt_id=row.prompt_id,
                label=label,
                chatbot_name=row.chatbot_name,
                trial=row.trial,
                temperature=row.temperature,
                timestamp=row.timestamp,
                response_token_count=row.response_token_count,
                response_token_count_empty=row.response_token_count_empty,
                response_token_count_nonempty=row.response_token_count_nonempty,
                data_lengths=row.data_lengths[:n],
                time_diffs=row.time_diffs[:n],
            )
            feat = row_to_feature_dict(prefix_row)
            vector = [float(feat[c]) for c in feature_cols]
            if split == "val":
                val_features[frac].append(vector)
            else:
                test_features[frac].append(vector)

        if split == "val":
            y_val.append(label)
        else:
            y_test.append(label)

    if not y_val or not y_test:
        raise RuntimeError("No val/test rows were collected for early-observation analysis.")

    y_val_np = np.array(y_val, dtype=int)
    y_test_np = np.array(y_test, dtype=int)

    results: List[Dict[str, object]] = []
    for frac in FRACTIONS:
        X_val = np.array(val_features[frac], dtype=float)
        X_test = np.array(test_features[frac], dtype=float)

        if len(X_val) != len(y_val_np) or len(X_test) != len(y_test_np):
            raise RuntimeError("Feature/label length mismatch while building prefix datasets.")

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        t_best = find_threshold_for_best_f1(y_val_np, val_scores)

        val_eval = _evaluate(y_val_np, val_scores, t_best, config.training.precision_at_k)
        test_eval = _evaluate(y_test_np, test_scores, t_best, config.training.precision_at_k)

        results.append(
            {
                "fraction": float(frac),
                "fraction_pct": int(round(frac * 100)),
                "val_threshold_best_f1": float(t_best),
                "val": val_eval,
                "test": test_eval,
            }
        )

    reference = tabular_metrics["models"]["xgboost"]["threshold_best_f1"]
    payload = {
        "dataset_path": dataset_path,
        "tabular_metrics_path": args.tabular_metrics,
        "model_path": args.model_path,
        "model_name": "xgboost",
        "fractions": FRACTIONS,
        "n_val": int(len(y_val_np)),
        "n_test": int(len(y_test_np)),
        "n_rows_kept": int(kept),
        "load_stats": load_stats,
        "results": results,
        "full_stream_reference": {
            "auprc": float(reference["auprc"]),
            "f1": float(reference["f1"]),
            "accuracy": float(reference["accuracy"]),
            "threshold": float(reference["threshold"]),
        },
        "env_versions": print_env_versions(),
    }
    save_json(args.output_json, payload)

    csv_rows: List[Dict[str, float]] = []
    for row in results:
        csv_rows.append(
            {
                "fraction_pct": float(row["fraction_pct"]),
                "val_threshold_best_f1": float(row["val_threshold_best_f1"]),
                "test_auprc": float(row["test"]["auprc"]),
                "test_f1": float(row["test"]["f1"]),
                "test_accuracy": float(row["test"]["accuracy"]),
                "test_precision": float(row["test"]["precision"]),
                "test_recall": float(row["test"]["recall"]),
            }
        )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(args.output_csv, index=False)

    print(f"Early-observation metrics saved: {args.output_json}")
    print(f"Early-observation summary csv saved: {args.output_csv}")
    print(f"Rows kept (val+test): {kept} | val={len(y_val_np)} test={len(y_test_np)}")


if __name__ == "__main__":
    main()
