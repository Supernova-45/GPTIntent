"""Cross-validation for tabular models."""

from __future__ import annotations

import glob
from pathlib import Path
import sys
from typing import Dict

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _build_models(seed: int):
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


def _mean_std(values: list[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def main() -> None:
    config_path = "artifacts/runs/data2_full_fast_lstm/runtime_config.json"
    dataset_path = ""
    folds = 5
    selected_models = ["logreg", "svm_rbf", "random_forest", "xgboost", "lightgbm"]
    output_path = "artifacts/runs/data2_full_fast_lstm/reports/group_cv_metrics.json"

    config = load_config(config_path)
    set_seed(config.training.random_seed)

    if not dataset_path:
        candidates = sorted(glob.glob(f"{config.paths.dataset_dir}/openrouter_intent_features.*"))
        if not candidates:
            raise FileNotFoundError(
                "Could not find dataset artifacts/datasets/openrouter_intent_features.*; run build_dataset.py first"
            )
        dataset_path = candidates[0]

    df = load_tabular_table(dataset_path)
    feature_cols = feature_columns_from_df(df)
    validate_feature_columns(feature_cols)

    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    groups = df["prompt_id"].values

    all_models = _build_models(config.training.random_seed)
    models = {name: model for name, model in all_models.items() if name in selected_models}
    if not models:
        raise ValueError(f"No valid models selected. Available: {', '.join(sorted(all_models.keys()))}")

    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.training.random_seed,
    )

    reports: Dict[str, object] = {}
    for model_name, model in models.items():
        fold_reports = []
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            classes, counts = np.unique(y_train, return_counts=True)
            class_weight = {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}
            sample_weight = np.array([class_weight[int(label)] for label in y_train], dtype=float)

            fit_kwargs = {}
            if model_name in {"logreg", "svm_rbf"}:
                fit_kwargs["model__sample_weight"] = sample_weight
            else:
                fit_kwargs["sample_weight"] = sample_weight

            model.fit(X_train, y_train, **fit_kwargs)

            y_score = model.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)

            fold_reports.append(
                {
                    "auprc": float(average_precision_score(y_test, y_score)),
                    "roc_auc": float(roc_auc_score(y_test, y_score)),
                    "f1_at_0.5": float(f1_score(y_test, y_pred)),
                    "accuracy_at_0.5": float(accuracy_score(y_test, y_pred)),
                    "precision_at_0.5": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall_at_0.5": float(recall_score(y_test, y_pred, zero_division=0)),
                }
            )

        reports[model_name] = {
            "folds": fold_reports,
            "summary": {
                "auprc": _mean_std([f["auprc"] for f in fold_reports]),
                "roc_auc": _mean_std([f["roc_auc"] for f in fold_reports]),
                "f1_at_0.5": _mean_std([f["f1_at_0.5"] for f in fold_reports]),
                "accuracy_at_0.5": _mean_std([f["accuracy_at_0.5"] for f in fold_reports]),
                "precision_at_0.5": _mean_std([f["precision_at_0.5"] for f in fold_reports]),
                "recall_at_0.5": _mean_std([f["recall_at_0.5"] for f in fold_reports]),
            },
        }

    ensure_dir(str(Path(output_path).parent))
    save_json(
        output_path,
        {
            "dataset_path": dataset_path,
            "folds": folds,
            "models": list(models.keys()),
            "feature_count": len(feature_cols),
            "reports": reports,
            "env_versions": print_env_versions(),
        },
    )
    print(f"Group CV metrics saved: {output_path}")


if __name__ == "__main__":
    main()
