#!/usr/bin/env python3
"""Run feature-family ablations on tabular dataset."""

from __future__ import annotations

import argparse
from typing import Dict, List
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_dependencies():
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "numpy, pandas, and scikit-learn are required. Install with: pip install numpy pandas scikit-learn"
        ) from ex


def _select_feature_family(columns: List[str], family: str) -> List[str]:
    size_cols = [c for c in columns if c.startswith("size_")]
    time_cols = [c for c in columns if c.startswith("time_")]
    token_cols = [c for c in columns if c.startswith("response_token_count")]
    meta_cols = [c for c in columns if c in {"temperature"}]

    if family == "size_only":
        return size_cols
    if family == "timing_only":
        return time_cols
    if family == "token_only":
        return token_cols
    if family == "timing_plus_size":
        return size_cols + time_cols
    if family == "no_token_counts":
        return [c for c in columns if c not in token_cols]
    if family == "all":
        return columns
    if family == "metadata_only":
        return meta_cols

    raise ValueError(f"Unknown ablation family: {family}")


def _run_single(df, features: List[str], seed: int, precision_ks, bootstrap_samples: int):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from analysis.common import (
        bootstrap_metric_ci,
        find_threshold_for_best_f1,
        precision_at_k,
    )

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[features].values
    y_train = train_df["label"].values.astype(int)
    X_val = val_df[features].values
    y_val = val_df["label"].values.astype(int)
    X_test = test_df[features].values
    y_test = test_df["label"].values.astype(int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

    val_scores = pipe.predict_proba(X_val)[:, 1]
    threshold = find_threshold_for_best_f1(y_val, val_scores)

    test_scores = pipe.predict_proba(X_test)[:, 1]
    y_pred = (test_scores >= threshold).astype(int)

    auprc = float(average_precision_score(y_test, test_scores))
    f1 = float(f1_score(y_test, y_pred))
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    p_at_k = {f"p@{int(k*100)}": float(precision_at_k(y_test, test_scores, k)) for k in precision_ks}

    auprc_ci = bootstrap_metric_ci(
        y_test,
        test_scores,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    return {
        "n_features": len(features),
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision_at_k": p_at_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    _require_dependencies()

    import glob

    from analysis.common import (
        ensure_dir,
        feature_columns_from_df,
        load_config,
        load_tabular_table,
        print_env_versions,
        save_json,
        set_seed,
        validate_feature_columns,
    )

    config = load_config(args.config)
    set_seed(config.training.random_seed)
    ensure_dir(config.paths.reports_dir)

    dataset_path = args.dataset
    if not dataset_path:
        candidates = sorted(glob.glob(f"{config.paths.dataset_dir}/openrouter_intent_features.*"))
        if not candidates:
            raise FileNotFoundError("Could not find dataset artifacts/datasets/openrouter_intent_features.*; run 01_build_dataset.py first")
        dataset_path = candidates[0]

    df = load_tabular_table(dataset_path)
    all_features = feature_columns_from_df(df)
    validate_feature_columns(all_features)

    families = [
        "all",
        "timing_only",
        "size_only",
        "token_only",
        "timing_plus_size",
        "no_token_counts",
    ]

    reports: Dict[str, object] = {}
    for fam in families:
        feats = _select_feature_family(all_features, fam)
        reports[fam] = _run_single(
            df,
            feats,
            seed=config.training.random_seed,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
        )

    out = f"{config.paths.reports_dir}/ablation_metrics.json"
    save_json(
        out,
        {
            "dataset_path": dataset_path,
            "families": reports,
            "env_versions": print_env_versions(),
        },
    )
    print(f"Ablation metrics saved: {out}")


if __name__ == "__main__":
    main()
