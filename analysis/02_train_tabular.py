#!/usr/bin/env python3
"""Train tabular intent models and write evaluation metrics."""

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple
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


def _build_models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    # Optional models
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lightgbm"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            random_state=seed,
            class_weight="balanced",
        )
    except Exception:
        pass

    return models


def _evaluate(
    y_true,
    y_score,
    threshold: float,
    precision_ks,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
    )

    from analysis.common import bootstrap_metric_ci, precision_at_k

    y_pred = (np.array(y_score) >= threshold).astype(int)
    auprc = float(average_precision_score(y_true, y_score))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    p_at_k = {f"p@{int(k*100)}": float(precision_at_k(y_true, y_score, k)) for k in precision_ks}

    auprc_ci = bootstrap_metric_ci(
        y_true,
        y_score,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    f1_ci = bootstrap_metric_ci(
        y_true,
        y_score,
        lambda yt, ys: float(f1_score(yt, (ys >= threshold).astype(int))),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    return {
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1": f1,
        "f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision_at_k": p_at_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular baselines")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional explicit path to dataset parquet/csv. If omitted, uses artifacts/datasets/openrouter_intent_features.*",
    )
    args = parser.parse_args()

    _require_dependencies()

    import numpy as np
    from sklearn.dummy import DummyClassifier

    from analysis.common import (
        ensure_dir,
        feature_columns_from_df,
        find_threshold_for_best_f1,
        find_threshold_for_target_precision,
        load_config,
        load_tabular_table,
        print_env_versions,
        save_json,
        set_seed,
        validate_feature_columns,
    )

    config = load_config(args.config)
    set_seed(config.training.random_seed)
    ensure_dir(config.paths.models_dir)
    ensure_dir(config.paths.reports_dir)

    dataset_path = args.dataset
    if not dataset_path:
        import glob

        candidates = sorted(glob.glob(f"{config.paths.dataset_dir}/openrouter_intent_features.*"))
        if not candidates:
            raise FileNotFoundError("Could not find dataset artifacts/datasets/openrouter_intent_features.*; run 01_build_dataset.py first")
        dataset_path = candidates[0]

    df = load_tabular_table(dataset_path)
    feature_cols = feature_columns_from_df(df)
    validate_feature_columns(feature_cols)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["label"].values.astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["label"].values.astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].values.astype(int)

    # Natural distribution eval; weighted training
    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    model_reports: Dict[str, object] = {}

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_scores = dummy.predict_proba(X_test)[:, 1]
    model_reports["dummy_prior"] = _evaluate(
        y_test,
        dummy_scores,
        threshold=0.5,
        precision_ks=config.training.precision_at_k,
        bootstrap_samples=config.training.bootstrap_samples,
        seed=config.training.random_seed,
    )

    models = _build_models(config.training.random_seed)
    for name, model in models.items():
        fit_kwargs = {}
        if name in {"logreg", "svm_linear", "svm_rbf", "random_forest", "xgboost", "lightgbm"}:
            # Pass sample weights when estimator supports it.
            if name in {"logreg", "svm_linear", "svm_rbf"}:
                fit_kwargs["model__sample_weight"] = sample_weight
            else:
                fit_kwargs["sample_weight"] = sample_weight

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        t_best_f1 = find_threshold_for_best_f1(y_val, val_scores)
        t_high_prec = find_threshold_for_target_precision(y_val, val_scores, target_precision=0.90)

        test_scores = model.predict_proba(X_test)[:, 1]

        report = {
            "threshold_best_f1": _evaluate(
                y_test,
                test_scores,
                threshold=t_best_f1,
                precision_ks=config.training.precision_at_k,
                bootstrap_samples=config.training.bootstrap_samples,
                seed=config.training.random_seed,
            ),
            "threshold_target_precision_0.90": _evaluate(
                y_test,
                test_scores,
                threshold=t_high_prec,
                precision_ks=config.training.precision_at_k,
                bootstrap_samples=config.training.bootstrap_samples,
                seed=config.training.random_seed,
            ),
        }
        model_reports[name] = report

        # Persist model where possible
        try:
            import joblib  # type: ignore

            model_path = f"{config.paths.models_dir}/{name}.joblib"
            joblib.dump(model, model_path)
        except Exception:
            pass

    payload = {
        "dataset_path": dataset_path,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "class_weight_train": class_weight,
        "models": model_reports,
        "env_versions": print_env_versions(),
    }

    metrics_path = f"{config.paths.reports_dir}/tabular_metrics.json"
    save_json(metrics_path, payload)

    print(f"Tabular metrics saved: {metrics_path}")
    print("Models:", ", ".join(model_reports.keys()))


if __name__ == "__main__":
    main()
