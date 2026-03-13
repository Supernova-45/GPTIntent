#!/usr/bin/env python3
"""Run a sweep of LSTM improvement methods and rank results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _full_experiments() -> List[Dict[str, object]]:
    return [
        {
            "name": "baseline",
            "args": [],
            "notes": "Original baseline settings.",
        },
        {
            "name": "log1p_norm",
            "args": ["--log1p-inputs", "--normalize-inputs"],
            "notes": "Stabilize skewed scales with log and z-score normalization.",
        },
        {
            "name": "bi_mean_pool",
            "args": [
                "--log1p-inputs",
                "--normalize-inputs",
                "--bidirectional",
                "--pooling",
                "mean",
                "--hidden-size",
                "96",
                "--num-layers",
                "2",
                "--dropout",
                "0.2",
            ],
            "notes": "Bidirectional context with smoother sequence pooling.",
        },
        {
            "name": "bi_attention_reg",
            "args": [
                "--log1p-inputs",
                "--normalize-inputs",
                "--bidirectional",
                "--pooling",
                "attention",
                "--hidden-size",
                "96",
                "--num-layers",
                "2",
                "--dropout",
                "0.3",
                "--scheduler",
                "plateau",
                "--early-stop-patience",
                "2",
                "--min-epochs",
                "4",
                "--grad-clip",
                "1.0",
                "--weight-decay",
                "1e-4",
            ],
            "notes": "Stronger model with validation-driven optimization controls.",
        },
    ]


def _fast_experiments() -> List[Dict[str, object]]:
    return [
        {
            "name": "baseline_fast",
            "args": [
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "2",
            ],
            "notes": "Lean baseline with early stopping.",
        },
        {
            "name": "log1p_norm_fast",
            "args": [
                "--log1p-inputs",
                "--normalize-inputs",
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "2",
            ],
            "notes": "Fast normalized variant.",
        },
        {
            "name": "bi_attention_reg_fast",
            "args": [
                "--log1p-inputs",
                "--normalize-inputs",
                "--bidirectional",
                "--pooling",
                "attention",
                "--hidden-size",
                "64",
                "--num-layers",
                "1",
                "--dropout",
                "0.2",
                "--scheduler",
                "plateau",
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "2",
                "--grad-clip",
                "1.0",
                "--weight-decay",
                "1e-4",
            ],
            "notes": "Fast regularized bidirectional attention variant.",
        },
    ]


def _tiny_experiments() -> List[Dict[str, object]]:
    return [
        {
            "name": "baseline_tiny",
            "args": [
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "1",
            ],
            "notes": "Very small budget baseline sanity check.",
        },
        {
            "name": "log1p_norm_tiny",
            "args": [
                "--log1p-inputs",
                "--normalize-inputs",
                "--early-stop-patience",
                "1",
                "--min-epochs",
                "1",
            ],
            "notes": "Very small budget normalized variant.",
        },
    ]


def _experiments_for_profile(profile: str) -> List[Dict[str, object]]:
    if profile == "full":
        return _full_experiments()
    if profile == "fast":
        return _fast_experiments()
    if profile == "tiny":
        return _tiny_experiments()
    raise ValueError(f"Unknown profile: {profile}")


def _resolved_hparams(args):
    defaults = {
        "full": {
            "epochs": 10,
            "batch_size": 64,
            "max_len": 1024,
            "train_limit": 0,
            "val_limit": 0,
            "test_limit": 0,
        },
        "fast": {
            "epochs": 3,
            "batch_size": 128,
            "max_len": 256,
            "train_limit": 6000,
            "val_limit": 1500,
            "test_limit": 1500,
        },
        "tiny": {
            "epochs": 2,
            "batch_size": 128,
            "max_len": 256,
            "train_limit": 4000,
            "val_limit": 1000,
            "test_limit": 1000,
        },
    }[args.profile]
    return {
        "epochs": args.epochs if args.epochs > 0 else defaults["epochs"],
        "batch_size": args.batch_size if args.batch_size > 0 else defaults["batch_size"],
        "max_len": args.max_len if args.max_len > 0 else defaults["max_len"],
        "train_limit": args.train_limit if args.train_limit > 0 else defaults["train_limit"],
        "val_limit": args.val_limit if args.val_limit > 0 else defaults["val_limit"],
        "test_limit": args.test_limit if args.test_limit > 0 else defaults["test_limit"],
    }


def _extract_result(payload: Dict[str, object]) -> Dict[str, float]:
    lstm = payload["lstm"]
    best_f1 = lstm["threshold_best_f1"]
    target_p = lstm["threshold_target_precision_0.90"]
    return {
        "auprc": float(best_f1["auprc"]),
        "f1": float(best_f1["f1"]),
        "accuracy": float(best_f1["accuracy"]),
        "f1_at_p90_threshold": float(target_p["f1"]),
        "best_val_auprc": float(lstm.get("best_val_auprc", 0.0)),
        "best_epoch": float(lstm.get("best_epoch", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LSTM improvement methods")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument("--profile", choices=["full", "fast", "tiny"], default="fast")
    parser.add_argument("--epochs", type=int, default=0, help="0 uses the profile default.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses the profile default.")
    parser.add_argument("--max-len", type=int, default=0, help="0 uses the profile default.")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional cap on training rows per experiment.")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional cap on validation rows per experiment.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional cap on test rows per experiment.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeout-minutes", type=int, default=0, help="Optional timeout per experiment.")
    parser.add_argument("--stop-after", type=int, default=0, help="Optional cap on number of experiments to run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop sweep immediately if one experiment fails.")
    parser.add_argument("--output", default="artifacts/reports/lstm_method_sweep.json")
    args = parser.parse_args()

    from analysis.common import ensure_dir, save_json

    ensure_dir("artifacts/reports")
    ensure_dir("artifacts/models")
    ensure_dir("artifacts/reports/lstm_sweep")

    resolved = _resolved_hparams(args)
    experiments = _experiments_for_profile(args.profile)
    if args.stop_after > 0:
        experiments = experiments[: args.stop_after]

    print(
        f"[sweep] profile={args.profile} experiments={len(experiments)} "
        f"epochs={resolved['epochs']} batch_size={resolved['batch_size']} "
        f"max_len={resolved['max_len']} device={args.device}",
        flush=True,
    )

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    sweep_start = time.time()
    total = len(experiments)
    for i, exp in enumerate(experiments, start=1):
        exp_start = time.time()
        name = str(exp["name"])
        report_path = f"artifacts/reports/lstm_sweep/{name}.json"
        model_path = f"artifacts/models/lstm_{name}.pt"
        cmd = [
            sys.executable,
            "analysis/03_train_lstm.py",
            "--config",
            args.config,
            "--epochs",
            str(resolved["epochs"]),
            "--batch-size",
            str(resolved["batch_size"]),
            "--device",
            args.device,
            "--max-len",
            str(resolved["max_len"]),
            "--report-path",
            report_path,
            "--model-path",
            model_path,
        ] + list(exp["args"])
        if resolved["train_limit"] > 0:
            cmd += ["--train-limit", str(resolved["train_limit"])]
        if resolved["val_limit"] > 0:
            cmd += ["--val-limit", str(resolved["val_limit"])]
        if resolved["test_limit"] > 0:
            cmd += ["--test-limit", str(resolved["test_limit"])]

        print(f"[sweep] ({i}/{total}) starting {name}", flush=True)
        print("[sweep] cmd:", " ".join(cmd), flush=True)
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=(args.timeout_minutes * 60) if args.timeout_minutes > 0 else None,
            )
        except subprocess.TimeoutExpired:
            failures.append(
                {
                    "name": name,
                    "status": "timeout",
                    "timeout_minutes": args.timeout_minutes,
                    "report_path": report_path,
                    "model_path": model_path,
                    "args": exp["args"],
                }
            )
            print(f"[sweep] timeout in {name} after {args.timeout_minutes} minutes", flush=True)
            if args.fail_fast:
                break
            continue
        except subprocess.CalledProcessError as ex:
            failures.append(
                {
                    "name": name,
                    "status": "failed",
                    "return_code": int(ex.returncode),
                    "report_path": report_path,
                    "model_path": model_path,
                    "args": exp["args"],
                }
            )
            print(f"[sweep] failed {name} with return code {ex.returncode}", flush=True)
            if args.fail_fast:
                break
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        summary = _extract_result(payload)
        results.append(
            {
                "name": name,
                "notes": exp["notes"],
                "args": exp["args"],
                "report_path": report_path,
                "model_path": model_path,
                "elapsed_sec": float(time.time() - exp_start),
                "metrics": summary,
            }
        )
        print(
            f"[sweep] ({i}/{total}) done {name} auprc={summary['auprc']:.4f} "
            f"f1={summary['f1']:.4f} elapsed_s={time.time()-exp_start:.1f}"
        , flush=True)

    ranked = sorted(results, key=lambda x: (x["metrics"]["auprc"], x["metrics"]["f1"]), reverse=True)
    out_payload = {
        "config": {
            "config_path": args.config,
            "profile": args.profile,
            "epochs": resolved["epochs"],
            "batch_size": resolved["batch_size"],
            "max_len": resolved["max_len"],
            "train_limit": resolved["train_limit"],
            "val_limit": resolved["val_limit"],
            "test_limit": resolved["test_limit"],
            "device": args.device,
            "timeout_minutes": args.timeout_minutes,
        },
        "ranking": ranked,
        "failures": failures,
        "best_method": ranked[0]["name"] if ranked else None,
        "elapsed_sec": float(time.time() - sweep_start),
    }
    save_json(args.output, out_payload)
    print(f"Method sweep saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
