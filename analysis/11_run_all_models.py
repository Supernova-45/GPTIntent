#!/usr/bin/env python3
"""Run end-to-end evaluation: tabular baselines + LSTM + blending."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_tabular(metrics_payload: Dict[str, object]) -> Dict[str, object]:
    models = metrics_payload["models"]
    ranked = []
    for name, report in models.items():
        if name == "dummy_prior":
            continue
        m = report["threshold_best_f1"]
        ranked.append(
            {
                "model": name,
                "auprc": float(m["auprc"]),
                "roc_auc": float(m.get("roc_auc", 0.0)),
                "f1": float(m["f1"]),
                "accuracy": float(m["accuracy"]),
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "threshold": float(m["threshold"]),
            }
        )
    ranked.sort(key=lambda x: (x["auprc"], x["f1"]), reverse=True)
    return ranked[0] if ranked else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines + LSTM + blend on a dataset/raw shard set.")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument(
        "--data-glob",
        default="",
        help="Override raw shard glob/path for dataset build and LSTM (e.g., data/shard*/...json).",
    )
    parser.add_argument("--prompts-path", default="", help="Optional override for prompts json.")
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional prebuilt tabular dataset path (.csv/.parquet). If set, dataset build is skipped.",
    )
    parser.add_argument("--run-name", default="full_eval")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--lstm-profile", choices=["baseline", "fast"], default="baseline")
    parser.add_argument("--lstm-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--tabular-models",
        default="logreg,random_forest,xgboost,lightgbm",
        help="Comma-separated tabular models for analysis/02_train_tabular.py.",
    )
    parser.add_argument("--blend-models", default="xgboost,lightgbm,random_forest")
    parser.add_argument("--blend-weight-step", type=float, default=0.05)
    args = parser.parse_args()

    from analysis.common import ensure_dir, load_config, save_json

    run_dir = Path(args.output_root) / args.run_name
    dataset_dir = run_dir / "datasets"
    reports_dir = run_dir / "reports"
    models_dir = run_dir / "models"
    pred_dir = reports_dir / "predictions"
    ensure_dir(str(dataset_dir))
    ensure_dir(str(reports_dir))
    ensure_dir(str(models_dir))
    ensure_dir(str(pred_dir))

    cfg = load_config(args.config)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["paths"]["dataset_dir"] = str(dataset_dir)
    cfg_dict["paths"]["reports_dir"] = str(reports_dir)
    cfg_dict["paths"]["models_dir"] = str(models_dir)
    if args.data_glob:
        cfg_dict["paths"]["data_glob"] = args.data_glob
    if args.prompts_path:
        cfg_dict["paths"]["prompts_path"] = args.prompts_path

    runtime_config = run_dir / "runtime_config.json"
    save_json(str(runtime_config), cfg_dict)

    dataset_path = args.dataset
    if not dataset_path:
        _run([sys.executable, "analysis/01_build_dataset.py", "--config", str(runtime_config)])
        qc_path = reports_dir / "dataset_qc.json"
        qc = _load_json(str(qc_path))
        dataset_path = str(qc["dataset_path"])
        if not os.path.isabs(dataset_path):
            dataset_path = str((REPO_ROOT / dataset_path).resolve())

    _run(
        [
            sys.executable,
            "analysis/02_train_tabular.py",
            "--config",
            str(runtime_config),
            "--dataset",
            str(dataset_path),
            "--predictions-dir",
            str(pred_dir),
            "--models",
            args.tabular_models,
        ]
    )

    lstm_metrics_path = reports_dir / "lstm_metrics.json"
    if not args.skip_lstm:
        if args.dataset and (not args.data_glob):
            raise ValueError(
                "Custom --dataset was provided but no --data-glob. "
                "Pass matching raw shard path/glob for LSTM or use --skip-lstm."
            )
        lstm_cmd = [
            sys.executable,
            "analysis/03_train_lstm.py",
            "--config",
            str(runtime_config),
            "--device",
            args.lstm_device,
            "--report-path",
            str(lstm_metrics_path),
            "--predictions-path",
            str(pred_dir / "lstm_predictions.json"),
        ]
        if args.lstm_profile == "fast":
            lstm_cmd += [
                "--epochs",
                "4",
                "--batch-size",
                "128",
                "--max-len",
                "256",
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "2",
                "--train-limit",
                "6000",
                "--val-limit",
                "1500",
                "--test-limit",
                "1500",
            ]
        _run(lstm_cmd)

    blend_models = [m.strip() for m in args.blend_models.split(",") if m.strip()]
    available_blend_models = [m for m in blend_models if (pred_dir / f"{m}_predictions.json").exists()]
    blend_path = reports_dir / "blend_metrics.json"
    blend_ran = False
    if len(available_blend_models) >= 2:
        _run(
            [
                sys.executable,
                "analysis/10_blend_models.py",
                "--pred-dir",
                str(pred_dir),
                "--models",
                ",".join(available_blend_models),
                "--weight-step",
                str(args.blend_weight_step),
                "--output",
                str(blend_path),
            ]
        )
        blend_ran = True
    else:
        print(
            f"[warn] Skipping blend: need >=2 available model prediction files; found {available_blend_models}",
            flush=True,
        )

    tabular_metrics = _load_json(str(reports_dir / "tabular_metrics.json"))
    best_tabular = _best_tabular(tabular_metrics)

    summary: Dict[str, object] = {
        "run_name": args.run_name,
        "runtime_config": str(runtime_config),
        "dataset_path": str(dataset_path),
        "paths": {
            "run_dir": str(run_dir),
            "reports_dir": str(reports_dir),
            "models_dir": str(models_dir),
            "predictions_dir": str(pred_dir),
            "tabular_metrics": str(reports_dir / "tabular_metrics.json"),
            "lstm_metrics": str(lstm_metrics_path) if lstm_metrics_path.exists() else None,
            "blend_metrics": str(blend_path) if blend_ran else None,
        },
        "best_tabular": best_tabular,
    }

    if lstm_metrics_path.exists():
        lstm_payload = _load_json(str(lstm_metrics_path))
        summary["lstm"] = lstm_payload["lstm"]["threshold_best_f1"]

    if blend_ran and blend_path.exists():
        blend_payload = _load_json(str(blend_path))
        summary["blend"] = {
            "models": blend_payload["models"],
            "weights": blend_payload["best_blend"]["weights"],
            "test_metrics": blend_payload["best_blend"]["test_metrics"],
            "deltas_vs_best_single": blend_payload["deltas_vs_best_single"],
        }

    summary_path = reports_dir / "full_eval_summary.json"
    save_json(str(summary_path), summary)
    print(f"Full evaluation summary saved: {summary_path}", flush=True)
    print("Best tabular:", best_tabular, flush=True)
    if "blend" in summary:
        print("Blend test metrics:", summary["blend"]["test_metrics"], flush=True)
    if "lstm" in summary:
        print("LSTM test metrics:", summary["lstm"], flush=True)


if __name__ == "__main__":
    main()
