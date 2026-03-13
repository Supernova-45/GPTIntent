"""Run feature-family and strict leakage ablations on the tabular dataset."""

from __future__ import annotations

import glob
from pathlib import Path
import sys
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import (
    bootstrap_metric_ci,
    ensure_dir,
    feature_columns_from_df,
    find_threshold_for_best_f1,
    load_config,
    load_tabular_table,
    precision_at_k,
    print_env_versions,
    save_json,
    set_seed,
    validate_feature_columns,
)


def _evaluate(y_true, y_score, threshold: float, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
    y_true_np = np.asarray(y_true, dtype=int)
    y_score_np = np.asarray(y_score, dtype=float)
    y_pred = (y_score_np >= threshold).astype(int)

    auprc = float(average_precision_score(y_true_np, y_score_np))
    f1 = float(f1_score(y_true_np, y_pred))
    p_at_k = {f"p@{int(k * 100)}": float(precision_at_k(y_true_np, y_score_np, k)) for k in precision_ks}

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
    }


def _build_logreg(seed: int):
    return Pipeline(
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


def _select_feature_family(columns: list[str], family: str) -> list[str]:
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


def _select_strict_feature_family(all_features: list[str], family: str) -> list[str]:
    feature_set = set(all_features)
    size_cols = sorted(c for c in all_features if c.startswith("size_"))
    time_cols = sorted(c for c in all_features if c.startswith("time_"))
    duration_cols = sorted(
        c
        for c in time_cols
        if c in {"time_mean", "time_min", "time_max"} or c.startswith("time_q")
    )

    remove_no_trial = {"trial"}
    remove_count_volume = {"packet_count"}
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
        raise ValueError(f"Unknown strict leakage family: {family}")
    drops = family_to_drop[family]
    return [c for c in all_features if c in feature_set and c not in drops]


def _run_single(df, features: list[str], seed: int, precision_ks, bootstrap_samples: int):
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[features].values
    y_train = train_df["label"].values.astype(int)
    X_val = val_df[features].values
    y_val = val_df["label"].values.astype(int)
    X_test = test_df[features].values
    y_test = test_df["label"].values.astype(int)

    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    model = _build_logreg(seed)
    model.fit(X_train, y_train, model__sample_weight=sample_weight)

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = find_threshold_for_best_f1(y_val, val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]

    return {
        "n_features": int(len(features)),
        "features": features,
        "test_metrics": _evaluate(
            y_test,
            test_scores,
            threshold=threshold,
            precision_ks=precision_ks,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def main() -> None:
    config_path = "artifacts/runs/data2_full_fast_lstm/runtime_config.json"
    dataset_path = ""
    family_ablations = [
        "all",
        "timing_only",
        "size_only",
        "token_only",
        "timing_plus_size",
        "no_token_counts",
    ]
    strict_leakage_ablations = [
        "all",
        "timing_only",
        "no_trial",
        "no_packet_count_total_bytes",
        "no_length_surrogates",
        "no_capture_duration",
        "strict_timing_signal",
    ]

    config = load_config(config_path)
    set_seed(config.training.random_seed)
    ensure_dir(config.paths.reports_dir)

    if not dataset_path:
        candidates = sorted(glob.glob(f"{config.paths.dataset_dir}/openrouter_intent_features.*"))
        if not candidates:
            raise FileNotFoundError(
                "Could not find dataset artifacts/datasets/openrouter_intent_features.*; run build_dataset.py first"
            )
        dataset_path = candidates[0]

    df = load_tabular_table(dataset_path)
    all_features = feature_columns_from_df(df)
    validate_feature_columns(all_features)

    family_reports: Dict[str, object] = {}
    for family in family_ablations:
        features = _select_feature_family(all_features, family)
        if not features:
            continue
        family_reports[family] = _run_single(
            df,
            features,
            seed=config.training.random_seed,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
        )

    strict_reports: Dict[str, object] = {}
    for family in strict_leakage_ablations:
        features = _select_strict_feature_family(all_features, family)
        if not features:
            continue
        strict_reports[family] = _run_single(
            df,
            features,
            seed=config.training.random_seed,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
        )

    ablation_output_path = f"{config.paths.reports_dir}/ablation_metrics.json"
    strict_output_path = f"{config.paths.reports_dir}/strict_leakage_ablations.json"

    payload = {
        "dataset_path": dataset_path,
        "settings": {
            "seed": int(config.training.random_seed),
            "precision_ks": [float(k) for k in config.training.precision_at_k],
            "bootstrap_samples": int(config.training.bootstrap_samples),
            "model": "logreg",
        },
        "families": family_reports,
        "strict_leakage_families": strict_reports,
        "env_versions": print_env_versions(),
    }
    save_json(ablation_output_path, payload)
    save_json(
        strict_output_path,
        {
            "dataset_path": dataset_path,
            "settings": payload["settings"],
            "families": strict_reports,
            "env_versions": payload["env_versions"],
        },
    )

    print(f"Ablation metrics saved: {ablation_output_path}")
    print(f"Strict leakage ablations saved: {strict_output_path}")


if __name__ == "__main__":
    main()
