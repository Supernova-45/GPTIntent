"""Generate report figures and provider-comparison summaries from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
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
    roc_curve,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import (
    bootstrap_metric_ci,
    ensure_dir,
    precision_at_k,
    print_env_versions,
    save_json,
)


PREFERRED_MODEL_ORDER = [
    "logreg",
    "svm_linear",
    "svm_rbf",
    "random_forest",
    "xgboost",
    "lightgbm",
]

DISPLAY_NAME = {
    "logreg": "LogReg",
    "svm_linear": "SVM (Linear)",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "blend": "Blend",
    "lstm": "LSTM",
}

MODEL_COLOR = {
    "logreg": "#9ecae1",
    "svm_linear": "#9ecae1",
    "svm_rbf": "#6baed6",
    "random_forest": "#4292c6",
    "xgboost": "#2171b5",
    "lightgbm": "#084594",
    "blend": "#31a354",
    "lstm": "#f28e2b",
}


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _confusion_from_counts(cm: list[list[int]]):
    arr = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return arr, arr / row_sums


def _plot_confusion(cm: list[list[int]], title: str, out_path: str) -> None:
    counts, norm = _confusion_from_counts(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

    labels = ["Benign (0)", "Malicious (1)"]
    ax.set_xticks([0, 1], labels=labels, rotation=15, ha="right")
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label", labelpad=12)
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(counts[i, j])}\n({norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.tight_layout(rect=(0.10, 0.0, 1.0, 1.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def _ordered_models(models: Dict[str, Dict[str, object]]) -> list[str]:
    comparable = [name for name, payload in models.items() if "threshold_best_f1" in payload]
    present = [name for name in PREFERRED_MODEL_ORDER if name in comparable]
    extras = sorted([name for name in comparable if name not in set(PREFERRED_MODEL_ORDER)])
    return present + extras


def _plot_model_auprc(
    models: Dict[str, Dict[str, object]],
    lstm_auprc: float,
    out_path: str,
    blend_auprc: float | None = None,
) -> None:
    ordered = _ordered_models(models)
    names = list(ordered)
    values = [float(models[name]["threshold_best_f1"]["auprc"]) for name in ordered]
    if blend_auprc is not None:
        names.append("blend")
        values.append(float(blend_auprc))
    names.append("lstm")
    values.append(float(lstm_auprc))
    labels = [DISPLAY_NAME.get(name, name) for name in names]
    colors = [MODEL_COLOR.get(name, "#4c4c4c") for name in names]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(labels, values, color=colors)
    y_min = max(0.0, min(values) - 0.08)
    y_max = min(1.0, max(values) + 0.04)
    if y_max - y_min < 0.15:
        y_min = max(0.0, y_max - 0.15)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Model Comparison (Test Split)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _extract_curve_points(y_true: list[int], y_score: list[float]):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return (recall.tolist(), precision.tolist()), (fpr.tolist(), tpr.tolist())


def _plot_curves(model_predictions: Dict[str, Dict[str, list[float]]], out_pr: str, out_roc: str) -> None:
    palette = {
        "logreg": MODEL_COLOR["logreg"],
        "svm_linear": MODEL_COLOR["svm_linear"],
        "svm_rbf": MODEL_COLOR["svm_rbf"],
        "random_forest": MODEL_COLOR["random_forest"],
        "xgboost": MODEL_COLOR["xgboost"],
        "lightgbm": MODEL_COLOR["lightgbm"],
        "blend": MODEL_COLOR["blend"],
    }

    fig_pr, ax_pr = plt.subplots(figsize=(4.8, 4.1))
    fig_roc, ax_roc = plt.subplots(figsize=(4.8, 4.1))

    for name, pred in model_predictions.items():
        y_true = pred["y_true"]
        y_score = pred["y_score"]
        (recall, precision), (fpr, tpr) = _extract_curve_points(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        color = palette.get(name, "#4c4c4c")
        display = DISPLAY_NAME.get(name, name)
        ax_pr.plot(recall, precision, lw=1.8, color=color, label=f"{display} (AUPRC={ap:.3f})")
        ax_roc.plot(fpr, tpr, lw=1.8, color=color, label=f"{display} (AUC={auc:.3f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (Test)")
    ax_pr.grid(alpha=0.25, linestyle="--")
    ax_pr.legend(fontsize=8, loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(out_pr, dpi=220)
    plt.close(fig_pr)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1.2)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves (Test)")
    ax_roc.grid(alpha=0.25, linestyle="--")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_roc, dpi=220)
    plt.close(fig_roc)


def _plot_early_observation(early_payload: Dict[str, object], out_path: str) -> None:
    rows = sorted(early_payload["results"], key=lambda row: float(row["fraction"]))
    x = [float(row["fraction_pct"]) for row in rows]
    auprc = [float(row["test"]["auprc"]) for row in rows]
    f1 = [float(row["test"]["f1"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(x, auprc, marker="o", lw=2.0, color="#1f77b4", label="AUPRC")
    ax.plot(x, f1, marker="s", lw=2.0, color="#d62728", label="F1")
    ax.set_xlabel("Observed stream fraction (%)")
    ax.set_ylabel("Test metric")
    ax.set_title("Early-Observation Attack Curve (XGBoost)")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_cv_stability(cv_payload: Dict[str, object], out_path: str) -> None:
    model_order = ["logreg", "svm_rbf", "random_forest", "xgboost", "lightgbm"]
    labels = ["LogReg", "SVM-RBF", "RF", "XGBoost", "LightGBM"]
    reports = cv_payload.get("reports", {})

    auprc_data = []
    f1_data = []
    use_labels = []
    for key, label in zip(model_order, labels):
        if key not in reports:
            continue
        folds = reports[key]["folds"]
        auprc_data.append([float(fold["auprc"]) for fold in folds])
        f1_data.append([float(fold["f1_at_0.5"]) for fold in folds])
        use_labels.append(label)

    if not auprc_data:
        return

    positions = np.arange(1, len(use_labels) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    metric_specs = [
        (axes[0], auprc_data, "Fold AUPRC", "AUPRC"),
        (axes[1], f1_data, "Fold F1@0.5", "F1@0.5"),
    ]

    for ax, data, title, ylabel in metric_specs:
        violin = ax.violinplot(data, positions=positions, widths=0.85, showmeans=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#7aa6d8")
            body.set_alpha(0.35)
            body.set_edgecolor("#3a6ea5")

        ax.boxplot(
            data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#1f1f1f", "linewidth": 1.3},
            boxprops={"facecolor": "#2c7fb8", "alpha": 0.65, "linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, labels=use_labels, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Group-CV Stability Across Tabular Models", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_precision_at_k_curve(pred_payloads: Dict[str, Dict[str, list[float]]], out_path: str) -> None:
    k_ratios = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    x = [k * 100 for k in k_ratios]

    palette = {
        "xgboost": MODEL_COLOR["xgboost"],
        "lightgbm": MODEL_COLOR["lightgbm"],
        "blend": MODEL_COLOR["blend"],
        "lstm": "#d95f0e",
    }
    labels = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "blend": "Blend", "lstm": "LSTM"}

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plotted = 0
    for model_name in ["xgboost", "lightgbm", "blend", "lstm"]:
        if model_name not in pred_payloads:
            continue
        y_true = np.array(pred_payloads[model_name]["y_true"], dtype=int)
        y_score = np.array(pred_payloads[model_name]["y_score"], dtype=float)
        values = [float(precision_at_k(y_true, y_score, k)) for k in k_ratios]
        ax.plot(
            x,
            values,
            marker="o",
            lw=2.0,
            color=palette.get(model_name, "#4c4c4c"),
            label=labels.get(model_name, model_name),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Flagged traffic (top % by model score)")
    ax.set_ylabel("Precision@k")
    ax.set_title("Operational Triage Curve (Top-k Precision)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(x), max(x))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _evaluate_with_uncertainty(y_true, y_score, threshold: float, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
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
    p_at_k_ci = {}
    for k in precision_ks:
        ci = bootstrap_metric_ci(
            y_true_np,
            y_score_np,
            lambda yt, ys, kk=k: float(precision_at_k(yt, ys, kk)),
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        p_at_k_ci[f"p@{int(k * 100)}"] = [float(ci[0]), float(ci[1])]

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


def _load_dataset_summary(
    dataset_name: str,
    tabular_metrics_path: str,
    lstm_metrics_path: str,
    blend_metrics_path: str,
    predictions_dir: str,
    model_names: list[str],
    precision_ks,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    tabular = _load_json(tabular_metrics_path)
    lstm_payload = _load_json(lstm_metrics_path)
    blend_payload = _load_json(blend_metrics_path) if Path(blend_metrics_path).exists() else {}
    pred_root = Path(predictions_dir)

    reports: Dict[str, object] = {}
    base_y_true = None
    base_scores_by_model: Dict[str, np.ndarray] = {}

    for model in model_names:
        pred_path = pred_root / f"{model}_predictions.json"
        if not pred_path.exists():
            continue
        pred = _load_json(str(pred_path))
        test = pred["test"]
        y_true = [int(x) for x in test["y_true"]]
        y_score = [float(x) for x in test["y_score"]]

        if base_y_true is None:
            base_y_true = y_true

        threshold = None
        if model in tabular.get("models", {}) and "threshold_best_f1" in tabular["models"][model]:
            threshold = float(tabular["models"][model]["threshold_best_f1"]["threshold"])
        elif model == "lstm":
            threshold = float(lstm_payload["lstm"]["threshold_best_f1"]["threshold"])
        if threshold is None:
            continue

        base_scores_by_model[model] = np.asarray(y_score, dtype=float)
        reports[model] = _evaluate_with_uncertainty(
            y_true,
            y_score,
            threshold=threshold,
            precision_ks=precision_ks,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    if "blend" in model_names and blend_payload.get("best_blend"):
        weights = blend_payload["best_blend"]["weights"]
        needed = [model for model in weights if model in base_scores_by_model]
        if len(needed) == len(weights) and base_y_true is not None:
            blend_scores = np.zeros_like(base_scores_by_model[needed[0]], dtype=float)
            for model, weight in weights.items():
                blend_scores += float(weight) * base_scores_by_model[model]
            threshold = float(blend_payload["best_blend"]["test_metrics"]["threshold"])
            reports["blend"] = _evaluate_with_uncertainty(
                base_y_true,
                blend_scores,
                threshold=threshold,
                precision_ks=precision_ks,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            reports["blend"]["weights"] = {model: float(weight) for model, weight in weights.items()}

    return {
        "dataset_name": dataset_name,
        "tabular_metrics_path": str(tabular_metrics_path),
        "lstm_metrics_path": str(lstm_metrics_path),
        "blend_metrics_path": str(blend_metrics_path),
        "predictions_dir": str(predictions_dir),
        "models": reports,
    }


def _plot_provider_drop(provider_a: Dict[str, object], provider_b: Dict[str, object], out_path: str) -> None:
    model_order = ["logreg", "random_forest", "xgboost", "lightgbm", "blend", "lstm"]
    present = [model for model in model_order if model in provider_a["models"] and model in provider_b["models"]]
    if not present:
        return

    x = np.arange(len(present))
    width = 0.38
    a_vals = [float(provider_a["models"][model]["auprc"]) for model in present]
    b_vals = [float(provider_b["models"][model]["auprc"]) for model in present]

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bars_a = ax.bar(x - width / 2, a_vals, width, label=provider_a["dataset_name"], color="#1f77b4")
    bars_b = ax.bar(x + width / 2, b_vals, width, label=provider_b["dataset_name"], color="#ff7f0e")

    ax.set_xticks(x, [DISPLAY_NAME.get(model, model) for model in present])
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Unpatched vs Patched Provider Performance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower left")

    for i, model in enumerate(present):
        drop = a_vals[i] - b_vals[i]
        y = max(a_vals[i], b_vals[i]) + 0.02
        ax.text(x[i], y, f"-{drop:.3f}", ha="center", va="bottom", fontsize=8, color="#444444")

    for bars in [bars_a, bars_b]:
        for bar in bars:
            value = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    tabular_metrics_path = "artifacts/runs/data2_full_fast_lstm/reports/tabular_metrics.json"
    lstm_metrics_path = "artifacts/runs/data2_full_fast_lstm/reports/lstm_metrics.json"
    predictions_dir = "artifacts/runs/data2_full_fast_lstm/reports/predictions"
    blend_metrics_path = "artifacts/runs/data2_full_fast_lstm/reports/blend_metrics.json"
    early_metrics_path = "artifacts/runs/data2_full_fast_lstm/reports/early_observation_metrics.json"
    group_cv_path = "artifacts/runs/data2_full_fast_lstm/reports/group_cv_metrics.json"
    output_dir = "artifacts/runs/data2_full_fast_lstm/reports/figures"

    provider_a_name = "OpenRouter (unpatched)"
    provider_a_tabular = "artifacts/reports/tabular_metrics.json"
    provider_a_lstm = "artifacts/reports/lstm_metrics.json"
    provider_a_blend = "artifacts/reports/blend_metrics.json"
    provider_a_predictions = "artifacts/reports/predictions"

    provider_b_name = "OpenAI (patched)"
    provider_b_tabular = tabular_metrics_path
    provider_b_lstm = lstm_metrics_path
    provider_b_blend = blend_metrics_path
    provider_b_predictions = predictions_dir

    provider_models = ["logreg", "random_forest", "xgboost", "lightgbm", "blend", "lstm"]
    provider_precision_ks = (0.01, 0.05, 0.10)
    provider_bootstrap_samples = 500
    provider_seed = 1337
    provider_summary_output_path = "artifacts/reports/provider_uncertainty_summary.json"
    provider_plot_output_path = f"{output_dir}/provider_drop_side_by_side.png"

    ensure_dir(output_dir)
    if not Path(provider_a_blend).exists():
        fallback_blend = Path("artifacts/reports/blend_metrics_xgb_lgbm_rf.json")
        if fallback_blend.exists():
            provider_a_blend = str(fallback_blend)

    tabular = _load_json(tabular_metrics_path)
    lstm_payload = _load_json(lstm_metrics_path)
    lstm_metrics = lstm_payload["lstm"]["threshold_best_f1"]
    blend_path = Path(blend_metrics_path)
    blend_payload = _load_json(str(blend_path)) if blend_path.exists() else {}
    blend_auprc = (
        float(blend_payload["best_blend"]["test_metrics"]["auprc"])
        if blend_payload.get("best_blend")
        else None
    )

    _plot_model_auprc(
        tabular["models"],
        lstm_auprc=float(lstm_metrics["auprc"]),
        blend_auprc=blend_auprc,
        out_path=f"{output_dir}/model_auprc_comparison.png",
    )

    tabular_models = tabular.get("models", {})
    comparable_models = {name: payload for name, payload in tabular_models.items() if "threshold_best_f1" in payload}
    best_tabular_name = max(
        comparable_models,
        key=lambda name: float(comparable_models[name]["threshold_best_f1"]["auprc"]),
    )
    _plot_confusion(
        comparable_models[best_tabular_name]["threshold_best_f1"]["confusion_matrix"],
        f"{DISPLAY_NAME.get(best_tabular_name, best_tabular_name)} Confusion Matrix (Test)",
        f"{output_dir}/best_tabular_confusion_matrix.png",
    )
    if "xgboost" in tabular_models:
        _plot_confusion(
            tabular_models["xgboost"]["threshold_best_f1"]["confusion_matrix"],
            "XGBoost Confusion Matrix (Test)",
            f"{output_dir}/xgboost_confusion_matrix.png",
        )
    if blend_payload.get("best_blend"):
        _plot_confusion(
            blend_payload["best_blend"]["test_metrics"]["confusion_matrix"],
            "Blend Confusion Matrix (Test)",
            f"{output_dir}/blend_confusion_matrix.png",
        )
    _plot_confusion(
        lstm_metrics["confusion_matrix"],
        "LSTM Confusion Matrix (Test)",
        f"{output_dir}/lstm_confusion_matrix.png",
    )

    model_predictions: Dict[str, Dict[str, list[float]]] = {}
    for name in _ordered_models(tabular_models):
        pred_path = Path(predictions_dir) / f"{name}_predictions.json"
        if not pred_path.exists():
            continue
        payload = _load_json(str(pred_path))["test"]
        model_predictions[name] = {
            "y_true": [int(x) for x in payload["y_true"]],
            "y_score": [float(x) for x in payload["y_score"]],
        }

    if blend_payload.get("best_blend"):
        weights = blend_payload["best_blend"]["weights"]
        if all(model in model_predictions for model in weights):
            y_true = model_predictions[next(iter(weights))]["y_true"]
            blend_scores = np.zeros(len(y_true), dtype=float)
            for model, weight in weights.items():
                blend_scores += float(weight) * np.array(model_predictions[model]["y_score"], dtype=float)
            model_predictions["blend"] = {
                "y_true": y_true,
                "y_score": blend_scores.tolist(),
            }

    if model_predictions:
        _plot_curves(
            model_predictions=model_predictions,
            out_pr=f"{output_dir}/tabular_pr_curves.png",
            out_roc=f"{output_dir}/tabular_roc_curves.png",
        )

    if Path(early_metrics_path).exists():
        early_payload = _load_json(early_metrics_path)
        _plot_early_observation(early_payload, f"{output_dir}/early_observation_curve.png")

    if Path(group_cv_path).exists():
        cv_payload = _load_json(group_cv_path)
        _plot_cv_stability(cv_payload, f"{output_dir}/cv_stability_violin.png")

    topk_models: Dict[str, Dict[str, list[float]]] = {}
    for name in ["xgboost", "lightgbm", "lstm"]:
        pred_path = Path(predictions_dir) / f"{name}_predictions.json"
        if not pred_path.exists():
            continue
        payload = _load_json(str(pred_path))
        if "test" not in payload:
            continue
        topk_models[name] = {
            "y_true": [int(x) for x in payload["test"]["y_true"]],
            "y_score": [float(x) for x in payload["test"]["y_score"]],
        }
    if "blend" in model_predictions:
        topk_models["blend"] = model_predictions["blend"]
    if topk_models:
        _plot_precision_at_k_curve(topk_models, f"{output_dir}/precision_at_k_curve.png")

    provider_inputs_exist = all(
        Path(path).exists()
        for path in [
            provider_a_tabular,
            provider_a_lstm,
            provider_a_predictions,
            provider_b_tabular,
            provider_b_lstm,
            provider_b_predictions,
        ]
    )
    if provider_inputs_exist:
        provider_a = _load_dataset_summary(
            dataset_name=provider_a_name,
            tabular_metrics_path=provider_a_tabular,
            lstm_metrics_path=provider_a_lstm,
            blend_metrics_path=provider_a_blend,
            predictions_dir=provider_a_predictions,
            model_names=provider_models,
            precision_ks=provider_precision_ks,
            bootstrap_samples=provider_bootstrap_samples,
            seed=provider_seed,
        )
        provider_b = _load_dataset_summary(
            dataset_name=provider_b_name,
            tabular_metrics_path=provider_b_tabular,
            lstm_metrics_path=provider_b_lstm,
            blend_metrics_path=provider_b_blend,
            predictions_dir=provider_b_predictions,
            model_names=provider_models,
            precision_ks=provider_precision_ks,
            bootstrap_samples=provider_bootstrap_samples,
            seed=provider_seed,
        )

        provider_drop = {}
        for model in provider_models:
            if model in provider_a["models"] and model in provider_b["models"]:
                provider_drop[model] = {
                    "dataset_a_auprc": float(provider_a["models"][model]["auprc"]),
                    "dataset_b_auprc": float(provider_b["models"][model]["auprc"]),
                    "delta_auprc_b_minus_a": float(
                        provider_b["models"][model]["auprc"] - provider_a["models"][model]["auprc"]
                    ),
                }

        save_json(
            provider_summary_output_path,
            {
                "settings": {
                    "models": provider_models,
                    "precision_ks": [float(k) for k in provider_precision_ks],
                    "bootstrap_samples": int(provider_bootstrap_samples),
                    "seed": int(provider_seed),
                },
                "providers": [provider_a, provider_b],
                "provider_drop": provider_drop,
                "output_plot": provider_plot_output_path,
                "env_versions": print_env_versions(),
            },
        )
        _plot_provider_drop(provider_a, provider_b, provider_plot_output_path)
        print(f"Provider uncertainty summary saved: {provider_summary_output_path}")
        print(f"Provider comparison plot saved: {provider_plot_output_path}")

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
