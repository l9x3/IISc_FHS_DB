# Federated Learning Framework for Fetal Heart Rate Prediction

## Overview

This module implements a **Federated Learning (FL)** system for fetal heart rate
prediction using simulated non-IID client data. The baseline model is a
4-layer Multi-Layer Perceptron (MLP) trained via FedAvg across 10 heterogeneous
clients.

---

## Architecture

### MLP Model (`client_model.py`)

| Layer | Dimension | Activation | Regularisation |
|-------|-----------|------------|----------------|
| Input | 20 features | — | — |
| FC-1 | 256 | ReLU | Dropout 0.3 |
| FC-2 | 128 | ReLU | Dropout 0.3 |
| FC-3 | 64 | ReLU | Dropout 0.3 |
| FC-4 | 32 | ReLU | — |
| Output | 1 | Linear | — |

- **Loss**: Mean Absolute Error (MAE)  
- **Optimiser**: SGD with momentum 0.9  
- **Default config**: batch size 512, learning rate 0.01

### Federated Learning Protocol

```
Server (global model)
   │
   ├─ broadcast global weights ──→ Client 1 … Client 10
   │                                    │
   │                               local SGD (5 epochs)
   │                                    │
   └─ aggregate via FedAvg ←────── updated weights
```

Aggregation follows **FedAvg** (McMahan et al., 2017): each client's contribution
is weighted by its local dataset size.

---

## Client Data Specifications

| Client | Samples | Dominant HR | Signal Quality |
|--------|---------|-------------|----------------|
| 1 | 70 000 | 140 BPM | Band-pass |
| 2 | 55 000 | 144 BPM | Wavelet |
| 3 | 48 000 | 135 BPM | EMD |
| 4 | 40 000 | 130 BPM | ICA |
| 5 | 30 000 | 146 BPM | Band-pass |
| 6 | 25 000 | 142 BPM | Wavelet |
| 7 | 20 000 | 148 BPM | EMD |
| 8 | 15 000 | 156 BPM | ICA |
| 9 | 10 000 | 150 BPM | Band-pass |
| 10 | 5 800 | 138 BPM | Wavelet |

Non-IID data distribution is simulated via:
- Different mean/std of target heart rate per client
- Different SNR / noise levels (signal-quality dependent)
- Heterogeneous feature distributions

---

## File Structure

```
.
├── config.yaml                    # FL hyper-parameter configuration
├── data_simulator.py              # Simulated non-IID data generation
├── client_model.py                # MLP model + FederatedClient class
├── server_aggregator.py           # FedAvg / FedMedian aggregation
├── federated_learning_framework.py # Main FL orchestration loop
├── visualization_plots.py         # All visualization routines
├── main_fl_experiment.py          # End-to-end experiment runner
└── results/
    └── federated/
        ├── metrics/
        │   └── fl_metrics.json    # Per-round metrics JSON
        └── plots/
            ├── 01_server_loss.png
            ├── 02_client_losses.png
            ├── 03_accuracy_improvement.png
            ├── 04_sample_distribution.png
            ├── 05_hr_distribution.png
            ├── 06_communication_efficiency.png
            ├── 07_federated_vs_centralized.png
            ├── 08_gradient_flow.png
            ├── 09_hyperparam_sweep.png
            ├── 10_data_heterogeneity.png
            └── 11_summary_dashboard.png
```

---

## Quick Start

```bash
# Install dependencies
pip install torch pyyaml seaborn matplotlib numpy

# Run full experiment (50 rounds + hyperparameter sweep)
python main_fl_experiment.py

# Quick smoke test (10 rounds, no sweep)
python main_fl_experiment.py --rounds 10 --no-sweep

# Custom configuration
python main_fl_experiment.py --config config.yaml --rounds 30 --sweep-rounds 5
```

---

## Generated Visualizations

| Plot | Description |
|------|-------------|
| `01_server_loss.png` | Global MAE per communication round |
| `02_client_losses.png` | Per-client MAE across all rounds |
| `03_accuracy_improvement.png` | Percentage MAE improvement vs. Round 1 |
| `04_sample_distribution.png` | Sample count per client (non-IID bar chart) |
| `05_hr_distribution.png` | Dominant heart rate per client |
| `06_communication_efficiency.png` | Rounds-to-convergence at various MAE thresholds |
| `07_federated_vs_centralized.png` | Federated vs. simulated centralised loss curves |
| `08_gradient_flow.png` | Round-to-round loss gradient per client |
| `09_hyperparam_sweep.png` | Heatmap of final MAE over batch-size × LR grid |
| `10_data_heterogeneity.png` | Dominant HR vs final MAE (bubble size ∝ samples) |
| `11_summary_dashboard.png` | 4-panel summary dashboard |

---

## Configuration (`config.yaml`)

Key parameters:

```yaml
federated_learning:
  num_clients: 10
  num_rounds: 50
  local_epochs: 5

model:
  input_dim: 20
  hidden_dims: [256, 128, 64, 32]
  dropout_rate: 0.3

training:
  batch_size: 512
  learning_rate: 0.01
  momentum: 0.9
```

---

## Results Summary

With the default configuration (50 rounds, batch 512, lr 0.01):

| Metric | Value |
|--------|-------|
| Initial Global MAE | ~12.8 BPM |
| Final Global MAE | ~1.8 BPM |
| Best sweep config | batch=256, lr=0.01 |

---

## References

- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Baseline MLP described in the IISc FHS research manuscript.
