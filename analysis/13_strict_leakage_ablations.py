#!/usr/bin/env python3
"""Run stronger leakage-control ablations on the tabular dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "numpy, pandas, and scikit-learn are required. Install with: "
            "pip install numpy pandas scikit-learn"
        ) from ex


def _evaluate(y_true, y_score, threshold, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
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

    from analysis.common import bootstrap_metric_ci, precision_at_k

    y_true_np = np.asarray(y_true, dtype=int)
    y_score_np = np.asarray(y_score, dtype=float)
    y_pred = (y_score_np >= threshold).astype(int)

    auprc = float(average_precision_score(y_true_np, y_score_np))
    f1 = float(f1_score(y_true_np, y_pred))
    p_at_k = {f"p@{int(k*100)}": float(precision_at_k(y_true_np, y_score_np, k)) for k in precision_ks}

    auprc_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    f1_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(f1_score(yt, (ys >= threshold).astype(int))),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    p_at_k_ci = {}
    for k in precision_ks:
        ci = bootstrap_metric_ci(
            y_true_np,
            y_score_np,
            lambda yt, ys, kk=k: float(precision_at_k(yt, ys, kk)),
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        p_at_k_ci[f"p@{int(k*100)}"] = [float(ci[0]), float(ci[1])]

    return {
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "roc_auc": float(roc_auc_score(y_true_np, y_score_np)),
        "f1": f1,
        "f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
        "precision_at_k": p_at_k,
        "precision_at_k_ci95": p_at_k_ci,
    }


def _select_features(all_features: List[str], family: str) -> List[str]:
    feature_set = set(all_features)
    size_cols = sorted(c for c in all_features if c.startswith("size_"))
    time_cols = sorted(c for c in all_features if c.startswith("time_"))
    duration_cols = sorted(
        c
        for c in time_cols
        if c in {"time_mean", "time_min", "time_max"} or c.startswith("time_q")
    )

    remove_no_trial = {"trial"}  # already excluded from the default table
    remove_count_volume = {"packet_count"}  # no explicit total-byte feature exists
    remove_length_surrogates = {"packet_count", "response_token_count_empty_pct"} | set(size_cols)
    remove_capture_duration = set(duration_cols)
    remove_strict = remove_no_trial | remove_count_volume | remove_length_surrogates | remove_capture_duration

    family_to_drop = {
        "all": set(),
        "timing_only": {c for c in all_features if not c.startswith("time_")},
        "no_trial": remove_no_trial,
        "no_packet_count_total_bytes": remove_count_volume,
        "no_length_surrogates": remove_length_surrogates,
        "no_capture_duration": remove_capture_duration,
        "strict_timing_signal": remove_strict,
    }
    if family not in family_to_drop:
        raise ValueError(f"Unknown family: {family}")
    drops = family_to_drop[family]
    return [c for c in all_features if c in feature_set and c not in drops]


def _fit_logreg(X_train, y_train, sample_weight, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict leakage-control ablations.")
    parser.add_argument(
        "--dataset",
        default="artifacts/datasets/openrouter_intent_features.csv",
        help="Tabular dataset path (Dataset A / OpenRouter by default).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--precision-ks", default="0.01,0.05,0.10")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--output",
        default="artifacts/reports/strict_leakage_ablations.json",
    )
    args = parser.parse_args()

    _require_dependencies()

    import numpy as np

    from analysis.common import (
        feature_columns_from_df,
        find_threshold_for_best_f1,
        load_tabular_table,
        print_env_versions,
        save_json,
        set_seed,
        validate_feature_columns,
    )

    set_seed(args.seed)
    precision_ks = tuple(float(x.strip()) for x in args.precision_ks.split(",") if x.strip())
    df = load_tabular_table(args.dataset)

    all_features = feature_columns_from_df(df)
    validate_feature_columns(all_features)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    families = [
        "all",
        "timing_only",
        "no_trial",
        "no_packet_count_total_bytes",
        "no_length_surrogates",
        "no_capture_duration",
        "strict_timing_signal",
    ]
    reports: Dict[str, object] = {}
    for fam in families:
        feats = _select_features(all_features, fam)
        if not feats:
            continue
        X_train = train_df[feats].values
        X_val = val_df[feats].values
        X_test = test_df[feats].values

        model = _fit_logreg(X_train, y_train, sample_weight=sample_weight, seed=args.seed)
        val_scores = model.predict_proba(X_val)[:, 1]
        threshold = find_threshold_for_best_f1(y_val, val_scores)
        test_scores = model.predict_proba(X_test)[:, 1]

        reports[fam] = {
            "n_features": int(len(feats)),
            "features": feats,
            "test_metrics": _evaluate(
                y_test,
                test_scores,
                threshold=threshold,
                precision_ks=precision_ks,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }

    payload = {
        "dataset_path": str(args.dataset),
        "settings": {
            "seed": int(args.seed),
            "precision_ks": [float(k) for k in precision_ks],
            "bootstrap_samples": int(args.bootstrap_samples),
            "model": "logreg",
        },
        "families": reports,
        "env_versions": print_env_versions(),
    }
    save_json(args.output, payload)
    print(f"Strict leakage ablations saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
