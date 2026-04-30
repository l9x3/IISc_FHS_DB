# Federated Learning for Fetal Heart Rate Prediction

A comprehensive federated learning system for fetal heart rate (FHR) prediction
using the IISc Fetal Heart Sound Database (IISc FHS DB).  The system enables
collaborative model training across 10 geographically distributed clients while
preserving data privacy through local-only data storage.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Guide](#configuration-guide)
6. [Client Specifications](#client-specifications)
7. [Local Model Architecture](#local-model-architecture)
8. [Federated Aggregation](#federated-aggregation)
9. [Evaluation](#evaluation)
10. [Results Interpretation](#results-interpretation)
11. [Heterogeneity Handling](#heterogeneity-handling)
12. [Privacy Considerations](#privacy-considerations)
13. [Reproducibility](#reproducibility)
14. [Project Structure](#project-structure)

---

## Project Overview

This project implements a federated learning (FL) pipeline in which 10 clients,
each holding private fetal heart sound / ECG recordings preprocessed with a
different method, collaboratively train a shared multilayer perceptron (MLP)
without ever sharing raw data.

**Key statistics:**

| Property | Value |
|---|---|
| Total feature windows | 409,852 |
| Clients | 10 |
| Heart rate range | 130–156 BPM |
| Preprocessing types | 4 (band-pass, wavelet, EMD, ICA) |
| Feature dimension | 256 |
| Model architecture | MLP 256→128→64→32→1 |
| Loss function | MAE |
| Aggregation | Weighted FedAvg |

---

## System Architecture

```
┌─────────────────────────────────────────┐
│  Central Aggregation Server              │
│  (FedAvg – weighted by dataset size)     │
└────────────────┬────────────────────────┘
                 │  global model weights (no raw data)
    ┌────────────┼─────────────────────────┐
    │            │             ...         │
    ▼            ▼                         ▼
┌──────────┐ ┌──────────┐           ┌──────────┐
│ Client 1 │ │ Client 2 │    ...    │ Client 10│
│ 70 000 w │ │ 55 000 w │           │  5 852 w │
│ band-pass│ │ wavelet  │           │ band-pass│
│ HR=140   │ │ HR=144   │           │ HR=130   │
└──────────┘ └──────────┘           └──────────┘
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/l9x3/IISc_FHS_DB.git
cd IISc_FHS_DB

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Optional: install the package in editable mode
pip install -e .
```

---

## Quick Start

### Run the full simulation

```bash
python experiments/simulate_federated_learning.py
```

### Override configuration via CLI flags

```bash
python experiments/simulate_federated_learning.py \
    --config config/federated_config.yaml \
    --rounds 50 \
    --results results/
```

### Run the test suite

```bash
pytest tests/ -v
```

### Minimal programmatic example

```python
from src.federated_trainer import FederatedTrainer
from src.utils import load_config

cfg = load_config("config/federated_config.yaml")
# Fast demo: 5 rounds
cfg["federated"]["num_communication_rounds"] = 5

trainer = FederatedTrainer(cfg, results_dir="results/")
trainer.setup()
metrics = trainer.run()

print(f"Final global MAE: {metrics[-1]['global_mae']:.4f} BPM")
```

---

## Configuration Guide

Edit `config/federated_config.yaml` to change any setting:

```yaml
model:
  input_dim: 256          # MLP input dimension
  dropout_rate: 0.3       # dropout after each hidden layer

training:
  batch_size: 512         # mini-batch size
  learning_rate: 0.01     # SGD learning rate
  local_epochs: 5         # epochs per communication round

federated:
  num_communication_rounds: 100
  client_participation_rate: 1.0   # 1.0 = all clients every round
  aggregation_method: fedavg_weighted
  seed: 42

evaluation:
  kfold: 5
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
```

---

## Client Specifications

| Client | Windows | Heart Rate (BPM) | Signal Type |
|--------|---------|-----------------|-------------|
| 1 | 70,000 | 140 | Band-pass filtered |
| 2 | 55,000 | 144 | Wavelet filtered |
| 3 | 48,000 | 148 | EMD denoised |
| 4 | 42,000 | 152 | ICA denoised |
| 5 | 45,000 | 136 | Wavelet filtered |
| 6 | 38,000 | 132 | Band-pass filtered |
| 7 | 30,000 | 156 | EMD denoised |
| 8 | 28,000 | 150 | ICA denoised |
| 9 | 48,000 | 138 | Wavelet filtered |
| 10 | 5,852 | 130 | Band-pass filtered |
| **Total** | **409,852** | **130–156** | **4 types** |

---

## Local Model Architecture

```
Input (256)
    │
    ├─ Linear(256 → 128) → ReLU → Dropout(0.3)
    ├─ Linear(128 →  64) → ReLU → Dropout(0.3)
    ├─ Linear( 64 →  32) → ReLU → Dropout(0.3)
    └─ Linear( 32 →   1)   [linear output, no activation]
```

**Loss function (MAE):**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} \left| f(\mathbf{x}_i; \boldsymbol{\theta}) - y_i \right|$$

**Final baseline hyperparameters:** batch size = 512, learning rate = 0.01

---

## Federated Aggregation

The server uses **weighted FedAvg**:

$$\theta_{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{n_{\text{total}}} \theta_k$$

where $n_k$ is client $k$'s dataset size.  This compensates for the highly
skewed data distribution (5,852–70,000 windows per client).

**Communication round cycle:**

1. Server broadcasts current global model weights to all clients.
2. Each client trains locally for `local_epochs` epochs using SGD + MAE loss.
3. Clients return updated weight state-dicts (raw data never leaves the client).
4. Server aggregates using weighted FedAvg.
5. Server broadcasts the new global model.

---

## Evaluation

### Per-round metrics

| Metric | Description |
|--------|-------------|
| `global_mae` | MAE of global model on all clients' test sets |
| `client_X_test_mae` | Per-client test MAE |
| `model_drift` | Frobenius-norm distance between consecutive global models |
| `wall_time_s` | Wall-clock time for the round |

### Group k-fold cross-validation

```python
from src.evaluation import FederatedEvaluator

ev = FederatedEvaluator(kfold=5)
results = ev.kfold_crossval(X_client, y_client)
print(results["summary"])
```

### Centralised vs. federated comparison

```python
metrics = ev.centralised_baseline(all_X, all_y, epochs=20)
print(f"Centralised MAE: {metrics['mae']:.4f}")
```

---

## Results Interpretation

After running the simulation, artefacts appear in `results/`:

```
results/
├── convergence_plots/
│   └── federated_convergence.png   # global + per-client MAE curves
├── metrics/
│   ├── round_metrics.csv           # tidy per-round table
│   ├── round_metrics.json          # raw round data
│   ├── summary.json                # experiment summary
│   └── comparison.json             # federated vs. centralised
└── model_checkpoints/
    └── global_model_final.pt       # final PyTorch state-dict
```

A decreasing `global_mae` curve confirms that the federated system is
converging.  The `model_drift` metric quantifies how much the global model
changes each round and should decrease as training stabilises.

---

## Heterogeneity Handling

| Heterogeneity Type | Manifestation | Mitigation |
|---|---|---|
| **Data Non-IID** | Different signal preprocessing per client | Each client's data distribution differs; weighted FedAvg adapts |
| **System Heterogeneity** | 5,852–70,000 windows per client | Weighted aggregation: $\propto n_k / n_{\text{total}}$ |
| **Statistical Heterogeneity** | HR varies 130–156 BPM | Per-client distribution shift in simulator; global model generalises |

---

## Privacy Considerations

* **Raw data never leaves a client.**  Only weight state-dicts are transmitted.
* Model weights alone carry some privacy risk (gradient inversion attacks).
* **Optional future work:**
  * Differential privacy noise injection before sharing weights.
  * Secure aggregation (cryptographic protocols).
  * Federated analytics with aggregated statistics only.

---

## Reproducibility

All random seeds are controlled through `config/federated_config.yaml`:

```yaml
federated:
  seed: 42
```

Setting the same seed guarantees identical synthetic data generation, model
initialisation, and training trajectories across runs.

---

## Project Structure

```
IISc_FHS_DB/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── federated_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_simulator.py       # Synthetic ECG/FHS signal generation
│   ├── model.py                # MLP 256→128→64→32→1 (PyTorch)
│   ├── client.py               # Local training (SGD, MAE)
│   ├── server.py               # Weighted FedAvg aggregation
│   ├── federated_trainer.py    # Orchestration loop
│   ├── evaluation.py           # Metrics & k-fold validation
│   ├── utils.py                # Signal preprocessing & helpers
│   └── logger.py               # Logging configuration
├── experiments/
│   └── simulate_federated_learning.py
├── tests/
│   ├── __init__.py
│   ├── test_data_simulator.py
│   ├── test_model.py
│   ├── test_client.py
│   ├── test_server.py
│   └── test_integration.py
├── results/                    # Generated artefacts (git-ignored)
│   ├── convergence_plots/
│   ├── metrics/
│   └── model_checkpoints/
└── deep_learning_fhs.py        # Existing centralised baseline (TF/Keras)
```
