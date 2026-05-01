"""
main_fl_experiment.py
=====================
End-to-end runner for the Federated Learning experiment.

Usage
-----
  python main_fl_experiment.py                   # full experiment
  python main_fl_experiment.py --rounds 10       # quick smoke test
  python main_fl_experiment.py --no-sweep        # skip hyperparameter sweep

The script:
  1. Loads config from config.yaml (or uses defaults).
  2. Runs the main FL experiment (FedAvg, 50 rounds, 10 clients).
  3. Optionally runs a hyperparameter sweep.
  4. Saves all visualizations to results/federated/plots/.
  5. Saves metrics JSON to results/federated/metrics/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML config; return empty dict if PyYAML unavailable."""
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        log.warning("PyYAML not installed — using default config.")
    except FileNotFoundError:
        log.warning("config.yaml not found — using defaults.")
    return {}


def _save_metrics(results: dict, out_dir: str) -> None:
    """Serialise results dict to JSON (convert numpy types first)."""
    os.makedirs(out_dir, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = os.path.join(out_dir, "fl_metrics.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    log.info("Metrics saved → %s", path)


def main(args: argparse.Namespace) -> None:
    from federated_learning_framework import FederatedLearning, run_hyperparam_sweep
    from visualization_plots import generate_all_plots

    config = _load_config(args.config)

    # Override num_rounds from CLI if provided
    if args.rounds is not None:
        config.setdefault("federated_learning", {})["num_rounds"] = args.rounds

    # ── Main FL experiment ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Federated Learning — Fetal Heart Rate Prediction")
    log.info("=" * 60)
    t0 = time.time()
    fl = FederatedLearning(config)
    results = fl.run()
    elapsed = time.time() - t0

    log.info("Experiment complete in %.1f s", elapsed)
    log.info("Final global MAE: %.4f BPM", results["server_loss"][-1])

    # ── Hyperparameter sweep ──────────────────────────────────────────────────
    sweep_results = None
    if not args.no_sweep:
        log.info("Running hyperparameter sweep …")
        sweep_cfg = config.get("hyperparameter_sweep", {})
        batch_sizes   = sweep_cfg.get("batch_sizes",   [128, 256, 512])
        learning_rates = sweep_cfg.get("learning_rates", [1e-4, 1e-3, 1e-2])
        sweep_results = run_hyperparam_sweep(
            base_config=config,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            num_rounds=args.sweep_rounds,
        )
        log.info("Sweep results:")
        for (bs, lr), mae in sorted(sweep_results.items()):
            log.info("  batch=%d  lr=%.0e  →  MAE=%.4f", bs, lr, mae)

    # ── Save metrics ──────────────────────────────────────────────────────────
    paths_cfg = config.get("paths", {})
    metrics_dir = paths_cfg.get("metrics_dir", "results/federated/metrics")
    _save_metrics(results, metrics_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = paths_cfg.get("plots_dir", "results/federated/plots")
    sweep_for_plot = None
    if sweep_results:
        # Convert tuple keys to str for JSON-friendliness; keep for plot function
        sweep_for_plot = {k: v for k, v in sweep_results.items()}
    generate_all_plots(results, sweep_results=sweep_for_plot, out_dir=plots_dir)

    log.info("All outputs saved to %s", plots_dir)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning experiment for fetal heart rate prediction"
    )
    parser.add_argument("--config",        default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--rounds",        type=int, default=None,
                        help="Override number of FL rounds")
    parser.add_argument("--no-sweep",      action="store_true",
                        help="Skip hyperparameter sweep")
    parser.add_argument("--sweep-rounds",  type=int, default=10,
                        help="Rounds per sweep configuration (default: 10)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
