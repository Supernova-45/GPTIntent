#!/usr/bin/env python3
"""Cross-provider transfer: train on one provider, test on the other."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
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


def _build_models(seed: int) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models: Dict[str, object] = {
        "logreg": Pipeline(
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
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

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


def _generate_weight_vectors(n_models: int, step: float) -> List[List[float]]:
    if n_models < 2:
        return [[1.0]]
    units = int(round(1.0 / step))
    out: List[List[float]] = []
    for combo in itertools.product(range(units + 1), repeat=n_models):
        if sum(combo) != units:
            continue
        out.append([c / units for c in combo])
    return out


def _find_best_threshold_for_f1(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_true, y_score)
    if len(t) == 0:
        return 0.5
    f1 = (2 * p * r) / (p + r + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def _metric_eval(
    y_true,
    y_score,
    threshold: float,
    precision_ks: Tuple[float, ...],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
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
    p_at_k = {
        f"p@{int(k*100)}": float(precision_at_k(y_true_np, y_score_np, k)) for k in precision_ks
    }

    auprc_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    f1_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(f1_score(yt, (ys >= threshold).astype(int))),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    p_at_k_ci = {}
    for k in precision_ks:
        ci = bootstrap_metric_ci(
            y_true_np,
            y_score_np,
            lambda yt, ys, kk=k: float(precision_at_k(yt, ys, kk)),
            n_bootstrap=n_bootstrap,
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


def _run_direction(
    source_name: str,
    source_df,
    target_name: str,
    target_df,
    model_names: List[str],
    blend_models: List[str],
    blend_weight_step: float,
    precision_ks: Tuple[float, ...],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    import numpy as np
    from sklearn.metrics import average_precision_score

    from analysis.common import feature_columns_from_df, validate_feature_columns

    source_train = source_df[source_df["split"] == "train"].copy()
    source_val = source_df[source_df["split"] == "val"].copy()
    target_test = target_df[target_df["split"] == "test"].copy()

    source_features = feature_columns_from_df(source_df)
    target_features = feature_columns_from_df(target_df)
    shared_features = [c for c in source_features if c in set(target_features)]
    validate_feature_columns(shared_features)

    X_train = source_train[shared_features].values
    y_train = source_train["label"].values.astype(int)
    X_val = source_val[shared_features].values
    y_val = source_val["label"].values.astype(int)
    X_test = target_test[shared_features].values
    y_test = target_test["label"].values.astype(int)

    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train], dtype=float)

    available = _build_models(seed)
    selected = [m for m in model_names if m in available]
    if not selected:
        raise ValueError("No requested models are available for transfer.")

    model_reports: Dict[str, object] = {}
    val_scores_by_model: Dict[str, np.ndarray] = {}
    test_scores_by_model: Dict[str, np.ndarray] = {}

    for name in selected:
        model = available[name]
        fit_kwargs = {}
        if name == "logreg":
            fit_kwargs["model__sample_weight"] = sample_weight
        else:
            fit_kwargs["sample_weight"] = sample_weight

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = _find_best_threshold_for_f1(y_val, val_scores)

        val_scores_by_model[name] = val_scores
        test_scores_by_model[name] = test_scores
        model_reports[name] = _metric_eval(
            y_test,
            test_scores,
            threshold=threshold,
            precision_ks=precision_ks,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    blend_report = None
    blend_available = [m for m in blend_models if m in val_scores_by_model]
    if len(blend_available) >= 2:
        candidates = _generate_weight_vectors(len(blend_available), blend_weight_step)
        best_weights = None
        best_val_auprc = -1.0
        for w in candidates:
            val_blend = np.zeros_like(y_val, dtype=float)
            for wi, m in zip(w, blend_available):
                val_blend += float(wi) * val_scores_by_model[m]
            val_auprc = float(average_precision_score(y_val, val_blend))
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_weights = w

        assert best_weights is not None
        val_blend = np.zeros_like(y_val, dtype=float)
        test_blend = np.zeros_like(y_test, dtype=float)
        for wi, m in zip(best_weights, blend_available):
            val_blend += float(wi) * val_scores_by_model[m]
            test_blend += float(wi) * test_scores_by_model[m]

        blend_threshold = _find_best_threshold_for_f1(y_val, val_blend)
        blend_eval = _metric_eval(
            y_test,
            test_blend,
            threshold=blend_threshold,
            precision_ks=precision_ks,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        blend_report = {
            "models": blend_available,
            "weights": {m: float(wi) for m, wi in zip(blend_available, best_weights)},
            "weight_step": float(blend_weight_step),
            "val_auprc": float(best_val_auprc),
            "num_weight_candidates": int(len(candidates)),
            "test": blend_eval,
        }

    return {
        "source_provider": source_name,
        "target_provider": target_name,
        "n_train_source": int(len(source_train)),
        "n_val_source": int(len(source_val)),
        "n_test_target": int(len(target_test)),
        "n_shared_features": int(len(shared_features)),
        "feature_columns": shared_features,
        "models": model_reports,
        "blend": blend_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-provider transfer experiments.")
    parser.add_argument(
        "--dataset-a-path",
        default="artifacts/datasets/openrouter_intent_features.csv",
        help="Dataset A path (usually OpenRouter/unpatched).",
    )
    parser.add_argument(
        "--dataset-b-path",
        default="artifacts/runs/data2_full_fast_lstm/datasets/openrouter_intent_features.csv",
        help="Dataset B path (usually OpenAI/patched).",
    )
    parser.add_argument("--dataset-a-name", default="openrouter_unpatched")
    parser.add_argument("--dataset-b-name", default="openai_patched")
    parser.add_argument("--models", default="logreg,random_forest,xgboost,lightgbm")
    parser.add_argument("--blend-models", default="xgboost,lightgbm,random_forest")
    parser.add_argument("--blend-weight-step", type=float, default=0.05)
    parser.add_argument("--precision-ks", default="0.01,0.05,0.10")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="artifacts/reports/cross_provider_transfer.json")
    args = parser.parse_args()

    _require_dependencies()

    from analysis.common import load_tabular_table, print_env_versions, save_json, set_seed

    set_seed(args.seed)
    dataset_a = load_tabular_table(args.dataset_a_path)
    dataset_b = load_tabular_table(args.dataset_b_path)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    blend_models = [m.strip() for m in args.blend_models.split(",") if m.strip()]
    precision_ks = tuple(float(x.strip()) for x in args.precision_ks.split(",") if x.strip())

    a_to_b = _run_direction(
        source_name=args.dataset_a_name,
        source_df=dataset_a,
        target_name=args.dataset_b_name,
        target_df=dataset_b,
        model_names=models,
        blend_models=blend_models,
        blend_weight_step=args.blend_weight_step,
        precision_ks=precision_ks,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    b_to_a = _run_direction(
        source_name=args.dataset_b_name,
        source_df=dataset_b,
        target_name=args.dataset_a_name,
        target_df=dataset_a,
        model_names=models,
        blend_models=blend_models,
        blend_weight_step=args.blend_weight_step,
        precision_ks=precision_ks,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    payload = {
        "dataset_a_path": str(args.dataset_a_path),
        "dataset_b_path": str(args.dataset_b_path),
        "settings": {
            "models": models,
            "blend_models": blend_models,
            "blend_weight_step": float(args.blend_weight_step),
            "precision_ks": [float(k) for k in precision_ks],
            "bootstrap_samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
        },
        "transfers": [a_to_b, b_to_a],
        "env_versions": print_env_versions(),
    }
    save_json(args.output, payload)
    print(f"Cross-provider transfer metrics saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
