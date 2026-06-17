"""
Federated learning package for fetal heart rate prediction.
"""

from .model import FHRPredictionMLP
from .data_simulator import ClientDataSimulator
from .client import FederatedClient
from .server import FederatedServer
from .federated_trainer import FederatedTrainer
from .evaluation import FederatedEvaluator
from .utils import (
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_mape,
    load_config,
    generate_all_figures,
    plot_convergence,
    plot_client_mae_trajectories,
    plot_model_drift,
    plot_client_data_distribution,
    plot_label_distributions,
    plot_federated_vs_centralised,
    plot_final_client_maes,
    plot_communication_cost,
    plot_round_times,
    plot_summary_dashboard,
)

__all__ = [
    "FHRPredictionMLP",
    "ClientDataSimulator",
    "FederatedClient",
    "FederatedServer",
    "FederatedTrainer",
    "FederatedEvaluator",
    "compute_mae",
    "compute_rmse",
    "compute_r2",
    "compute_mape",
    "load_config",
    "generate_all_figures",
    "plot_convergence",
    "plot_client_mae_trajectories",
    "plot_model_drift",
    "plot_client_data_distribution",
    "plot_label_distributions",
    "plot_federated_vs_centralised",
    "plot_final_client_maes",
    "plot_communication_cost",
    "plot_round_times",
    "plot_summary_dashboard",
]
