"""Train tabular models, blend predictions, and run cross-provider transfer."""

from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path
import sys
from typing import Dict

import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import (
    bootstrap_metric_ci,
    ensure_dir,
    feature_columns_from_df,
    find_threshold_for_best_f1,
    find_threshold_for_target_precision,
    load_config,
    load_tabular_table,
    precision_at_k,
    print_env_versions,
    save_json,
    set_seed,
    validate_feature_columns,
)


def _build_models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
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
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            random_state=seed,
            class_weight="balanced",
        ),
    }


def _evaluate(y_true, y_score, threshold: float, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

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
        "roc_auc": float(roc_auc_score(y_true_np, y_score_np)),
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1": f1,
        "f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
        "precision_at_k": p_at_k,
    }


def _resolve_dataset_path(dataset_dir: str, dataset_path: str) -> str:
    if dataset_path:
        return dataset_path
    candidates = sorted(glob.glob(f"{dataset_dir}/openrouter_intent_features.*"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find dataset artifacts/datasets/openrouter_intent_features.*; run build_dataset.py first"
        )
    return candidates[0]


def _save_predictions(predictions_dir: str, model_name: str, val_df, test_df, y_val, y_test, val_scores, test_scores) -> None:
    if not predictions_dir:
        return
    save_json(
        f"{predictions_dir}/{model_name}_predictions.json",
        {
            "model": model_name,
            "val": {
                "sample_id": val_df["sample_id"].tolist(),
                "prompt_id": val_df["prompt_id"].tolist(),
                "y_true": [int(x) for x in y_val.tolist()],
                "y_score": [float(x) for x in val_scores.tolist()],
            },
            "test": {
                "sample_id": test_df["sample_id"].tolist(),
                "prompt_id": test_df["prompt_id"].tolist(),
                "y_true": [int(x) for x in y_test.tolist()],
                "y_score": [float(x) for x in test_scores.tolist()],
            },
        },
    )


def _load_predictions(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align_split(payload: Dict[str, object], split: str):
    data = payload[split]
    sample_ids = [str(x) for x in data["sample_id"]]
    y_true = [int(x) for x in data["y_true"]]
    y_score = [float(x) for x in data["y_score"]]
    return sample_ids, y_true, y_score


def _eval_metrics_basic(y_true, y_score, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _generate_weight_vectors(n_models: int, step: float) -> list[list[float]]:
    if n_models < 2:
        return [[1.0]]
    units = int(round(1.0 / step))
    out = []
    for combo in itertools.product(range(units + 1), repeat=n_models):
        if sum(combo) != units:
            continue
        out.append([c / units for c in combo])
    return out


def _score_blend(scores_by_model: Dict[str, list[float] | np.ndarray], weights: list[float]) -> np.ndarray:
    score = np.zeros_like(np.asarray(next(iter(scores_by_model.values()))), dtype=float)
    for weight, (_, scores) in zip(weights, scores_by_model.items()):
        score += weight * np.asarray(scores, dtype=float)
    return score


def _fit_with_weights(model_name: str, model, X_train, y_train, sample_weight) -> None:
    if model_name in {"logreg", "svm_linear", "svm_rbf"}:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)


def _find_dataset_in_dir(dataset_dir: str) -> str:
    candidates = sorted(glob.glob(f"{dataset_dir}/openrouter_intent_features.*"))
    return candidates[0] if candidates else ""


def _run_blend(predictions_dir: str, models: list[str], weight_step: float, output_path: str) -> Dict[str, object]:
    if len(models) < 2:
        raise ValueError("Need at least 2 models to blend.")

    payloads = {}
    for model in models:
        path = Path(predictions_dir) / f"{model}_predictions.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        payloads[model] = _load_predictions(str(path))

    ref_model = models[0]
    ref_val_ids, y_val, _ = _align_split(payloads[ref_model], "val")
    ref_test_ids, y_test, _ = _align_split(payloads[ref_model], "test")

    val_scores_by_model: Dict[str, list[float]] = {}
    test_scores_by_model: Dict[str, list[float]] = {}
    for model in models:
        val_ids, y_val_model, val_scores = _align_split(payloads[model], "val")
        test_ids, y_test_model, test_scores = _align_split(payloads[model], "test")
        if val_ids != ref_val_ids or test_ids != ref_test_ids:
            raise ValueError(f"Sample ID ordering mismatch for model {model}.")
        if y_val_model != y_val or y_test_model != y_test:
            raise ValueError(f"Label mismatch for model {model}.")
        val_scores_by_model[model] = val_scores
        test_scores_by_model[model] = test_scores

    best_single_name = None
    best_single_val_auprc = -1.0
    best_single_test_metrics = None
    for model in models:
        val_auprc = _eval_metrics_basic(y_val, val_scores_by_model[model], threshold=0.5)["auprc"]
        if val_auprc > best_single_val_auprc:
            best_single_val_auprc = val_auprc
            best_single_name = model
            threshold = find_threshold_for_best_f1(y_val, val_scores_by_model[model])
            best_single_test_metrics = _eval_metrics_basic(y_test, test_scores_by_model[model], threshold=threshold)

    candidates = _generate_weight_vectors(len(models), weight_step)
    best = None
    for weights in candidates:
        val_blend = _score_blend({model: val_scores_by_model[model] for model in models}, weights)
        val_auprc = _eval_metrics_basic(y_val, val_blend, threshold=0.5)["auprc"]
        if best is None or val_auprc > best["val_auprc"]:
            best = {"weights": weights, "val_auprc": float(val_auprc)}

    if best is None or best_single_test_metrics is None:
        raise ValueError("Could not pick a blend configuration from the available predictions.")
    best_weights = best["weights"]
    val_blend = _score_blend({model: val_scores_by_model[model] for model in models}, best_weights)
    test_blend = _score_blend({model: test_scores_by_model[model] for model in models}, best_weights)
    blend_threshold = find_threshold_for_best_f1(y_val, val_blend)
    blend_test_metrics = _eval_metrics_basic(y_test, test_blend, threshold=blend_threshold)

    payload = {
        "models": models,
        "weight_step": float(weight_step),
        "best_single_model": best_single_name,
        "best_single_test_metrics": best_single_test_metrics,
        "best_blend": {
            "weights": {model: float(weight) for model, weight in zip(models, best_weights)},
            "val_auprc": float(best["val_auprc"]),
            "test_metrics": blend_test_metrics,
        },
        "deltas_vs_best_single": {
            metric: float(blend_test_metrics[metric] - best_single_test_metrics[metric])
            for metric in ["auprc", "roc_auc", "f1", "accuracy", "precision", "recall"]
        },
        "num_weight_candidates": int(len(candidates)),
    }
    save_json(output_path, payload)
    return payload


def _run_transfer_direction(
    source_name: str,
    source_df,
    target_name: str,
    target_df,
    model_names: list[str],
    blend_models: list[str],
    blend_weight_step: float,
    precision_ks,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    from sklearn.metrics import average_precision_score

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
    selected_models = [model for model in model_names if model in available]
    if not selected_models:
        raise ValueError("No requested models are available for transfer.")

    model_reports: Dict[str, object] = {}
    val_scores_by_model: Dict[str, np.ndarray] = {}
    test_scores_by_model: Dict[str, np.ndarray] = {}

    for model_name in selected_models:
        model = available[model_name]
        _fit_with_weights(model_name, model, X_train, y_train, sample_weight)

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = find_threshold_for_best_f1(y_val, val_scores)

        val_scores_by_model[model_name] = val_scores
        test_scores_by_model[model_name] = test_scores
        model_reports[model_name] = _evaluate(
            y_test,
            test_scores,
            threshold=threshold,
            precision_ks=precision_ks,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    blend_report = None
    blend_available = [model for model in blend_models if model in val_scores_by_model]
    if len(blend_available) >= 2:
        candidates = _generate_weight_vectors(len(blend_available), blend_weight_step)
        best_weights = None
        best_val_auprc = -1.0
        for weights in candidates:
            val_blend = np.zeros_like(y_val, dtype=float)
            for weight, model in zip(weights, blend_available):
                val_blend += float(weight) * val_scores_by_model[model]
            val_auprc = float(average_precision_score(y_val, val_blend))
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_weights = weights

        if best_weights is None:
            raise ValueError("Could not find blend weights for the transfer experiment.")
        val_blend = np.zeros_like(y_val, dtype=float)
        test_blend = np.zeros_like(y_test, dtype=float)
        for weight, model in zip(best_weights, blend_available):
            val_blend += float(weight) * val_scores_by_model[model]
            test_blend += float(weight) * test_scores_by_model[model]

        threshold = find_threshold_for_best_f1(y_val, val_blend)
        blend_report = {
            "models": blend_available,
            "weights": {model: float(weight) for model, weight in zip(blend_available, best_weights)},
            "weight_step": float(blend_weight_step),
            "val_auprc": float(best_val_auprc),
            "num_weight_candidates": int(len(candidates)),
            "test": _evaluate(
                y_test,
                test_blend,
                threshold=threshold,
                precision_ks=precision_ks,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
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


def _run_cross_provider_transfer(
    dataset_a_path: str,
    dataset_b_path: str,
    dataset_a_name: str,
    dataset_b_name: str,
    model_names: list[str],
    blend_models: list[str],
    blend_weight_step: float,
    precision_ks,
    bootstrap_samples: int,
    seed: int,
    output_path: str,
) -> Dict[str, object]:
    dataset_a = load_tabular_table(dataset_a_path)
    dataset_b = load_tabular_table(dataset_b_path)

    a_to_b = _run_transfer_direction(
        source_name=dataset_a_name,
        source_df=dataset_a,
        target_name=dataset_b_name,
        target_df=dataset_b,
        model_names=model_names,
        blend_models=blend_models,
        blend_weight_step=blend_weight_step,
        precision_ks=precision_ks,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    b_to_a = _run_transfer_direction(
        source_name=dataset_b_name,
        source_df=dataset_b,
        target_name=dataset_a_name,
        target_df=dataset_a,
        model_names=model_names,
        blend_models=blend_models,
        blend_weight_step=blend_weight_step,
        precision_ks=precision_ks,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    payload = {
        "dataset_a_path": str(dataset_a_path),
        "dataset_b_path": str(dataset_b_path),
        "settings": {
            "models": model_names,
            "blend_models": blend_models,
            "blend_weight_step": float(blend_weight_step),
            "precision_ks": [float(k) for k in precision_ks],
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
        },
        "transfers": [a_to_b, b_to_a],
        "env_versions": print_env_versions(),
    }
    save_json(output_path, payload)
    return payload


def main() -> None:
    config_path = "artifacts/runs/data2_full_fast_lstm/runtime_config.json"
    dataset_path = ""
    predictions_dir = "artifacts/runs/data2_full_fast_lstm/reports/predictions"
    selected_models: list[str] = []
    blend_models = ["xgboost", "lightgbm", "random_forest"]
    blend_weight_step = 0.05
    run_blend = True
    run_cross_provider_transfer = True
    reference_dataset_path = ""
    reference_dataset_dir = "artifacts/datasets"
    reference_dataset_name = "openrouter_unpatched"
    current_dataset_name = "openai_patched"
    cross_provider_output_path = "artifacts/reports/cross_provider_transfer.json"

    config = load_config(config_path)
    set_seed(config.training.random_seed)
    ensure_dir(config.paths.models_dir)
    ensure_dir(config.paths.reports_dir)
    ensure_dir(predictions_dir)

    dataset_path = _resolve_dataset_path(config.paths.dataset_dir, dataset_path)
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
    if selected_models:
        models = {name: model for name, model in models.items() if name in selected_models}
    if not models:
        raise ValueError("No valid models selected.")

    for model_name, model in models.items():
        _fit_with_weights(model_name, model, X_train, y_train, sample_weight)

        val_scores = model.predict_proba(X_val)[:, 1]
        threshold_best_f1 = find_threshold_for_best_f1(y_val, val_scores)
        threshold_high_precision = find_threshold_for_target_precision(y_val, val_scores, target_precision=0.90)
        test_scores = model.predict_proba(X_test)[:, 1]

        _save_predictions(predictions_dir, model_name, val_df, test_df, y_val, y_test, val_scores, test_scores)

        model_reports[model_name] = {
            "threshold_best_f1": _evaluate(
                y_test,
                test_scores,
                threshold=threshold_best_f1,
                precision_ks=config.training.precision_at_k,
                bootstrap_samples=config.training.bootstrap_samples,
                seed=config.training.random_seed,
            ),
            "threshold_target_precision_0.90": _evaluate(
                y_test,
                test_scores,
                threshold=threshold_high_precision,
                precision_ks=config.training.precision_at_k,
                bootstrap_samples=config.training.bootstrap_samples,
                seed=config.training.random_seed,
            ),
        }
        joblib.dump(model, f"{config.paths.models_dir}/{model_name}.joblib")

    metrics_path = f"{config.paths.reports_dir}/tabular_metrics.json"
    payload = {
        "dataset_path": dataset_path,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "class_weight_train": class_weight,
        "models": model_reports,
        "env_versions": print_env_versions(),
    }
    save_json(metrics_path, payload)

    blend_payload = None
    blend_output_path = f"{config.paths.reports_dir}/blend_metrics.json"
    available_blend_models = [model for model in blend_models if model in models]
    if run_blend and len(available_blend_models) >= 2:
        blend_payload = _run_blend(
            predictions_dir=predictions_dir,
            models=available_blend_models,
            weight_step=blend_weight_step,
            output_path=blend_output_path,
        )

    transfer_payload = None
    transfer_models = [model for model in ["logreg", "random_forest", "xgboost", "lightgbm"] if model in _build_models(config.training.random_seed)]
    if not reference_dataset_path:
        reference_dataset_path = _find_dataset_in_dir(reference_dataset_dir)
    if run_cross_provider_transfer and reference_dataset_path and Path(reference_dataset_path).exists():
        transfer_payload = _run_cross_provider_transfer(
            dataset_a_path=reference_dataset_path,
            dataset_b_path=dataset_path,
            dataset_a_name=reference_dataset_name,
            dataset_b_name=current_dataset_name,
            model_names=transfer_models,
            blend_models=[model for model in blend_models if model in transfer_models],
            blend_weight_step=blend_weight_step,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
            seed=config.training.random_seed,
            output_path=cross_provider_output_path,
        )

    print(f"Tabular metrics saved: {metrics_path}")
    if blend_payload is not None:
        print(f"Blend metrics saved: {blend_output_path}")
    if transfer_payload is not None:
        print(f"Cross-provider transfer metrics saved: {cross_provider_output_path}")
    print("Models:", ", ".join(model_reports.keys()))


if __name__ == "__main__":
    main()
