"""Grouped-CV hyperparameter tuning + seed ensembling for tabular models."""

from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Dict

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import (
    feature_columns_from_df,
    load_config,
    load_tabular_table,
    print_env_versions,
    save_json,
    validate_feature_columns,
)


def _build_model_xgb(seed: int, params: Dict[str, object]):
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_lambda=float(params["reg_lambda"]),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def _build_model_lgbm(seed: int, params: Dict[str, object]):
    return LGBMClassifier(
        n_estimators=int(params["n_estimators"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=float(params["reg_lambda"]),
        objective="binary",
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


def _param_grid(model_name: str, space: str) -> list[Dict[str, object]]:
    if model_name == "xgboost":
        if space == "fast":
            grid = {
                "n_estimators": [180, 260, 340],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.08],
                "subsample": [0.85, 1.0],
                "colsample_bytree": [0.85, 1.0],
                "min_child_weight": [1.0, 3.0],
                "reg_lambda": [1.0, 3.0],
            }
        else:
            grid = {
                "n_estimators": [250, 400, 600],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_weight": [1.0, 3.0, 5.0],
                "reg_lambda": [1.0, 3.0],
            }
    elif model_name == "lightgbm":
        if space == "fast":
            grid = {
                "n_estimators": [220, 320, 420],
                "num_leaves": [31, 63],
                "max_depth": [-1, 10],
                "learning_rate": [0.05, 0.08],
                "subsample": [0.85, 1.0],
                "colsample_bytree": [0.85, 1.0],
                "min_child_samples": [20, 40],
                "reg_lambda": [0.0, 1.0],
            }
        else:
            grid = {
                "n_estimators": [250, 400, 600],
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 8, 12],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_samples": [20, 40, 80],
                "reg_lambda": [0.0, 1.0, 3.0],
            }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    keys = sorted(grid.keys())
    values = [grid[key] for key in keys]
    return [{key: value for key, value in zip(keys, combo)} for combo in itertools.product(*values)]


def _hash_params(params: Dict[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    import hashlib
    return hashlib.sha1(payload).hexdigest()[:12]


def _fit_predict(model_name: str, seed: int, params: Dict[str, object], X_train, y_train, X_eval):
    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    if model_name == "xgboost":
        model = _build_model_xgb(seed=seed, params=params)
    else:
        model = _build_model_lgbm(seed=seed, params=params)

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_eval)[:, 1]


def _group_cv_score(
    model_name: str,
    params: Dict[str, object],
    seeds: list[int],
    X,
    y,
    groups,
    folds: int,
    cv_seed: int,
) -> Dict[str, float]:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cv_seed)

    auprcs = []
    aucs = []
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        val_scores_by_seed = []
        for seed in seeds:
            val_scores_by_seed.append(_fit_predict(model_name, seed, params, X_train, y_train, X_val))
        y_score = np.mean(np.vstack(val_scores_by_seed), axis=0)

        auprcs.append(float(average_precision_score(y_val, y_score)))
        aucs.append(float(roc_auc_score(y_val, y_score)))

    return {
        "cv_auprc_mean": float(np.mean(auprcs)),
        "cv_auprc_std": float(np.std(auprcs, ddof=1) if len(auprcs) > 1 else 0.0),
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_std": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0),
    }


def _evaluate_on_split(
    model_name: str,
    params: Dict[str, object],
    seeds: list[int],
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> Dict[str, object]:
    val_scores_by_seed = []
    test_scores_by_seed = []
    for seed in seeds:
        val_scores_by_seed.append(_fit_predict(model_name, seed, params, X_train, y_train, X_val))
        test_scores_by_seed.append(_fit_predict(model_name, seed, params, X_train, y_train, X_test))

    val_score = np.mean(np.vstack(val_scores_by_seed), axis=0)
    test_score = np.mean(np.vstack(test_scores_by_seed), axis=0)

    precision, recall, thresholds = precision_recall_curve(y_val, val_score)
    f1s = (2 * precision * recall) / (precision + recall + 1e-12)
    threshold = 0.5 if len(thresholds) == 0 else float(thresholds[int(np.argmax(f1s[:-1]))])

    y_pred = (test_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    return {
        "threshold": threshold,
        "auprc": float(average_precision_score(y_test, test_score)),
        "roc_auc": float(roc_auc_score(y_test, test_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    config_path = "artifacts/runs/data2_full_fast_lstm/runtime_config.json"
    dataset_path = ""
    model_name = "xgboost"
    folds = 3
    max_trials = 12
    seeds = [1337, 2024, 7]
    search_seed = 1337
    space = "fast"
    output_path = "artifacts/runs/data2_full_fast_lstm/reports/ensemble_tune_xgboost_fast.json"

    cfg = load_config(config_path)
    if not dataset_path:
        candidates = sorted(glob.glob(f"{cfg.paths.dataset_dir}/openrouter_intent_features.*"))
        if not candidates:
            raise FileNotFoundError("Dataset not found; run build_dataset.py first.")
        dataset_path = candidates[0]

    df = load_tabular_table(dataset_path)
    feature_cols = feature_columns_from_df(df)
    validate_feature_columns(feature_cols)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)
    g_train = train_df["prompt_id"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    g_trainval = np.concatenate([g_train, val_df["prompt_id"].values], axis=0)

    if not seeds:
        raise ValueError("At least one seed is required.")

    grid = _param_grid(model_name, space=space)
    rng = random.Random(search_seed)
    rng.shuffle(grid)
    trials = grid[:max_trials]

    print(
        f"[tune] model={model_name} space={space} folds={folds} "
        f"trials={len(trials)} seeds={seeds}",
        flush=True,
    )

    scored = []
    for i, params in enumerate(trials, start=1):
        cv_stats = _group_cv_score(
            model_name=model_name,
            params=params,
            seeds=seeds,
            X=X_trainval,
            y=y_trainval,
            groups=g_trainval,
            folds=folds,
            cv_seed=search_seed,
        )
        scored.append(
            {
                "trial": i,
                "params": params,
                "params_id": _hash_params(params),
                **cv_stats,
            }
        )
        print(
            f"[tune] {i}/{len(trials)} auprc={cv_stats['cv_auprc_mean']:.4f} "
            f"+/-{cv_stats['cv_auprc_std']:.4f} params={_hash_params(params)}",
            flush=True,
        )

    ranked = sorted(scored, key=lambda row: (row["cv_auprc_mean"], row["cv_roc_auc_mean"]), reverse=True)
    best = ranked[0]
    print(f"[tune] best params_id={best['params_id']} auprc={best['cv_auprc_mean']:.4f}", flush=True)

    split_eval = _evaluate_on_split(
        model_name=model_name,
        params=best["params"],
        seeds=seeds,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    payload = {
        "config": {
            "dataset_path": dataset_path,
            "model": model_name,
            "space": space,
            "folds": folds,
            "max_trials": max_trials,
            "seeds": seeds,
            "search_seed": search_seed,
        },
        "feature_count": len(feature_cols),
        "best_cv": best,
        "test_eval_best_cv_model": split_eval,
        "top5_cv": ranked[:5],
        "all_trials": ranked,
        "env_versions": print_env_versions(),
    }
    save_json(output_path, payload)
    print(f"Tuning report saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
