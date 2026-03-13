"""Run a sweep of LSTM improvement methods."""

from __future__ import annotations

import time
from pathlib import Path
import sys
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import ensure_dir, load_config, save_json
from analysis.lstm import run_lstm_experiment


def _full_experiments() -> list[Dict[str, object]]:
    return [
        {
            "name": "baseline",
            "settings": {},
            "notes": "Original baseline settings.",
        },
        {
            "name": "log1p_norm",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
            },
            "notes": "Stabilize skewed scales with log and z-score normalization.",
        },
        {
            "name": "bi_mean_pool",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
                "bidirectional": True,
                "pooling": "mean",
                "hidden_size": 96,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "notes": "Bidirectional context with smoother sequence pooling.",
        },
        {
            "name": "bi_attention_reg",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
                "bidirectional": True,
                "pooling": "attention",
                "hidden_size": 96,
                "num_layers": 2,
                "dropout": 0.3,
                "scheduler_name": "plateau",
                "early_stop_patience": 2,
                "min_epochs": 4,
                "grad_clip": 1.0,
                "weight_decay": 1e-4,
            },
            "notes": "Stronger model with validation-driven optimization controls.",
        },
    ]


def _fast_experiments() -> list[Dict[str, object]]:
    return [
        {
            "name": "baseline_fast",
            "settings": {
                "early_stop_patience": 1,
                "min_epochs": 2,
            },
            "notes": "Lean baseline with early stopping.",
        },
        {
            "name": "log1p_norm_fast",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
                "early_stop_patience": 1,
                "min_epochs": 2,
            },
            "notes": "Fast normalized variant.",
        },
        {
            "name": "bi_attention_reg_fast",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
                "bidirectional": True,
                "pooling": "attention",
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.2,
                "scheduler_name": "plateau",
                "early_stop_patience": 1,
                "min_epochs": 2,
                "grad_clip": 1.0,
                "weight_decay": 1e-4,
            },
            "notes": "Fast regularized bidirectional attention variant.",
        },
    ]


def _tiny_experiments() -> list[Dict[str, object]]:
    return [
        {
            "name": "baseline_tiny",
            "settings": {
                "early_stop_patience": 1,
                "min_epochs": 1,
            },
            "notes": "Very small budget baseline sanity check.",
        },
        {
            "name": "log1p_norm_tiny",
            "settings": {
                "log1p_inputs": True,
                "normalize_inputs": True,
                "early_stop_patience": 1,
                "min_epochs": 1,
            },
            "notes": "Very small budget normalized variant.",
        },
    ]


def _experiments_for_profile(profile: str) -> list[Dict[str, object]]:
    if profile == "full":
        return _full_experiments()
    if profile == "fast":
        return _fast_experiments()
    if profile == "tiny":
        return _tiny_experiments()
    raise ValueError(f"Unknown profile: {profile}")


def _resolved_hparams(
    profile: str,
    epochs: int,
    batch_size: int,
    max_len: int,
    train_limit: int,
    val_limit: int,
    test_limit: int,
) -> Dict[str, int]:
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
    }[profile]
    return {
        "epochs": epochs if epochs > 0 else defaults["epochs"],
        "batch_size": batch_size if batch_size > 0 else defaults["batch_size"],
        "max_len": max_len if max_len > 0 else defaults["max_len"],
        "train_limit": train_limit if train_limit > 0 else defaults["train_limit"],
        "val_limit": val_limit if val_limit > 0 else defaults["val_limit"],
        "test_limit": test_limit if test_limit > 0 else defaults["test_limit"],
    }


def _extract_result(payload: Dict[str, object]) -> Dict[str, float]:
    lstm = payload["lstm"]
    best_f1 = lstm["threshold_best_f1"]
    target_precision = lstm["threshold_target_precision_0.90"]
    return {
        "auprc": float(best_f1["auprc"]),
        "f1": float(best_f1["f1"]),
        "accuracy": float(best_f1["accuracy"]),
        "f1_at_p90_threshold": float(target_precision["f1"]),
        "best_val_auprc": float(lstm.get("best_val_auprc", 0.0)),
        "best_epoch": float(lstm.get("best_epoch", 0)),
    }


def main() -> None:
    config_path = "artifacts/runs/data2_full_fast_lstm/runtime_config.json"
    profile = "fast"
    epochs = 0
    batch_size = 0
    max_len = 0
    train_limit = 0
    val_limit = 0
    test_limit = 0
    device_name = "auto"
    stop_after = 0
    fail_fast = False
    output_path = ""

    config = load_config(config_path)
    reports_dir = Path(config.paths.reports_dir)
    models_dir = Path(config.paths.models_dir)
    sweep_dir = reports_dir / "lstm_sweep"

    ensure_dir(str(reports_dir))
    ensure_dir(str(models_dir))
    ensure_dir(str(sweep_dir))

    resolved = _resolved_hparams(profile, epochs, batch_size, max_len, train_limit, val_limit, test_limit)
    experiments = _experiments_for_profile(profile)
    if stop_after > 0:
        experiments = experiments[:stop_after]

    print(
        f"[sweep] profile={profile} experiments={len(experiments)} "
        f"epochs={resolved['epochs']} batch_size={resolved['batch_size']} "
        f"max_len={resolved['max_len']} device={device_name}",
        flush=True,
    )

    results = []
    failures = []
    sweep_start = time.time()
    total = len(experiments)
    for i, experiment in enumerate(experiments, start=1):
        exp_start = time.time()
        name = str(experiment["name"])
        report_path = str(sweep_dir / f"{name}.json")
        model_path = str(models_dir / f"lstm_{name}.pt")

        settings = {
            "config_path": config_path,
            "epochs": resolved["epochs"],
            "batch_size": resolved["batch_size"],
            "device_name": device_name,
            "max_len_override": resolved["max_len"],
            "train_limit": resolved["train_limit"],
            "val_limit": resolved["val_limit"],
            "test_limit": resolved["test_limit"],
            "report_path": report_path,
            "model_path": model_path,
            "predictions_path": "",
            **experiment["settings"],
        }

        print(f"[sweep] ({i}/{total}) starting {name}", flush=True)
        try:
            payload = run_lstm_experiment(**settings)
        except Exception as ex:
            failures.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": str(ex),
                    "report_path": report_path,
                    "model_path": model_path,
                    "settings": experiment["settings"],
                }
            )
            print(f"[sweep] failed {name}: {ex}", flush=True)
            if fail_fast:
                break
            continue

        summary = _extract_result(payload)
        results.append(
            {
                "name": name,
                "notes": experiment["notes"],
                "settings": experiment["settings"],
                "report_path": report_path,
                "model_path": model_path,
                "elapsed_sec": float(time.time() - exp_start),
                "metrics": summary,
            }
        )
        print(
            f"[sweep] ({i}/{total}) done {name} auprc={summary['auprc']:.4f} "
            f"f1={summary['f1']:.4f} elapsed_s={time.time() - exp_start:.1f}",
            flush=True,
        )

    ranked = sorted(results, key=lambda row: (row["metrics"]["auprc"], row["metrics"]["f1"]), reverse=True)
    if not output_path:
        output_path = str(reports_dir / f"lstm_method_sweep_{profile}.json")

    save_json(
        output_path,
        {
            "config": {
                "config_path": config_path,
                "profile": profile,
                "epochs": resolved["epochs"],
                "batch_size": resolved["batch_size"],
                "max_len": resolved["max_len"],
                "train_limit": resolved["train_limit"],
                "val_limit": resolved["val_limit"],
                "test_limit": resolved["test_limit"],
                "device": device_name,
            },
            "ranking": ranked,
            "failures": failures,
            "best_method": ranked[0]["name"] if ranked else None,
            "elapsed_sec": float(time.time() - sweep_start),
        },
    )
    print(f"Method sweep saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
