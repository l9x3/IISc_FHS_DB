"""
End-to-end federated learning simulation for fetal heart rate prediction.

Usage
-----
    python experiments/simulate_federated_learning.py

    # Override defaults via CLI flags:
    python experiments/simulate_federated_learning.py \
        --config config/federated_config.yaml \
        --rounds 50 \
        --results results/

Run from the repository root directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow imports from repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.evaluation import FederatedEvaluator
from src.federated_trainer import FederatedTrainer
from src.logger import get_logger
from src.utils import generate_all_figures, load_config, save_json

log = get_logger(__name__, log_dir="results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate federated learning for fetal heart rate prediction"
    )
    p.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "federated_config.yaml",
        ),
        help="Path to YAML configuration file",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override num_communication_rounds in config",
    )
    p.add_argument(
        "--results",
        default="results",
        help="Directory for output artefacts",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, e.g. 'cpu' or 'cuda:0'",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Federated Learning – Fetal Heart Rate Prediction")
    log.info("=" * 60)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg["federated"]["num_communication_rounds"] = args.rounds

    device = torch.device(args.device)
    results_dir = args.results
    os.makedirs(results_dir, exist_ok=True)

    num_rounds = cfg["federated"]["num_communication_rounds"]
    num_clients = len(cfg.get("clients", []))
    log.info("Rounds: %d | Clients: %d | Device: %s", num_rounds, num_clients, device)

    # ── Build & run federated trainer ─────────────────────────────────────────
    trainer = FederatedTrainer(cfg, results_dir=results_dir, device=device)
    trainer.setup()

    t0 = time.perf_counter()
    round_metrics = trainer.run()
    total_time = time.perf_counter() - t0

    log.info("Training complete in %.1f s (%.2f s / round)", total_time, total_time / num_rounds)

    # ── Evaluate & export ──────────────────────────────────────────────────────
    evaluator = FederatedEvaluator(results_dir=results_dir, kfold=5, device=device)

    csv_path = evaluator.export_round_metrics(round_metrics)
    log.info("Per-round CSV  → %s", csv_path)

    # Convergence summary
    global_maes = [m["global_mae"] for m in round_metrics]
    convergence_rate = evaluator.compute_convergence_rate(global_maes)

    comm_bytes = evaluator.estimate_communication_bytes(
        trainer.server.global_model,
        num_rounds=num_rounds,
        num_clients=num_clients,
    )

    final_mae = global_maes[-1] if global_maes else float("nan")
    log.info("Final global MAE : %.4f BPM", final_mae)
    log.info("Convergence rate : %.4f (last-5-round relative decrease)", convergence_rate)
    log.info(
        "Communication    : ~%.1f MB transferred total",
        comm_bytes / 1_048_576,
    )

    # Per-client final test MAE
    final_round = round_metrics[-1] if round_metrics else {}
    client_maes = final_round.get("client_test_maes", {})

    # Distribution stats
    dist_stats = [
        evaluator.client_data_stats(sim.client_id, sim.generate()[1])
        for sim in trainer.simulators
    ]

    summary = {
        "config": {
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "batch_size": cfg["training"]["batch_size"],
            "learning_rate": cfg["training"]["learning_rate"],
            "local_epochs": cfg["training"]["local_epochs"],
        },
        "results": {
            "final_global_mae": final_mae,
            "best_global_mae": min(global_maes) if global_maes else float("nan"),
            "best_round": int(np.argmin(global_maes) + 1) if global_maes else -1,
            "convergence_rate": convergence_rate,
            "total_train_time_s": total_time,
            "communication_bytes": comm_bytes,
        },
        "final_client_test_maes": client_maes,
        "data_distribution": dist_stats,
    }
    summary_path = evaluator.export_summary(summary)
    log.info("Summary JSON   → %s", summary_path)

    # ── Centralised baseline comparison ───────────────────────────────────────
    log.info("Computing centralised baseline …")
    all_X = [sim.generate()[0] for sim in trainer.simulators]
    all_y = [sim.generate()[1] for sim in trainer.simulators]

    train_cfg = cfg.get("training", {})
    centralised_metrics = evaluator.centralised_baseline(
        all_X,
        all_y,
        input_dim=cfg["model"].get("input_dim", 256),
        dropout_rate=cfg["model"].get("dropout_rate", 0.3),
        epochs=train_cfg.get("local_epochs", 5),
        batch_size=train_cfg.get("batch_size", 512),
        learning_rate=train_cfg.get("learning_rate", 0.01),
    )
    log.info("Centralised baseline MAE : %.4f BPM", centralised_metrics["mae"])
    log.info(
        "Federated vs centralised  : Δ MAE = %.4f BPM",
        final_mae - centralised_metrics["mae"],
    )

    comparison = {
        "federated_final_mae": final_mae,
        "centralised_mae": centralised_metrics["mae"],
        "centralised_metrics": centralised_metrics,
    }
    save_json(comparison, os.path.join(results_dir, "metrics", "comparison.json"))

    # ── Generate all figures ───────────────────────────────────────────────────
    log.info("Generating figures …")
    plots_dir = os.path.join(results_dir, "convergence_plots")

    # Collect per-client HR labels for violin plot (sample a small subset for speed)
    client_labels: dict = {}
    for sim in trainer.simulators:
        _, y_all = sim.generate()
        client_labels[sim.client_id] = y_all

    client_specs = cfg.get("clients", [])
    bytes_per_round = evaluator.estimate_communication_bytes(
        trainer.server.global_model,
        num_rounds=1,
        num_clients=num_clients,
    )

    figure_paths = generate_all_figures(
        round_metrics=round_metrics,
        client_specs=client_specs,
        client_labels=client_labels,
        federated_mae=final_mae,
        centralised_mae=centralised_metrics["mae"],
        output_dir=plots_dir,
        bytes_per_round=float(bytes_per_round),
    )
    for fp in figure_paths:
        log.info("  Figure saved → %s", fp)

    log.info("=" * 60)
    log.info("Simulation finished.  Artefacts in: %s", results_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
