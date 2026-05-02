# IISc Fetal Heart Sound Dataset (IISc_FHS_DB)

A collection of fetal heart sound recordings from the Indian Institute of Science
(IISc), with annotated fetal heart rates (FHR, in bpm).

---

## Repository structure

```
IISc_FHS_DB/
├── dataset/
│   ├── Records.csv               # Subject metadata and FHR labels (bpm)
│   └── subject_XX.wav            # Raw fetal heart sound recordings
├── deep_learning_fhs.py          # Single-site deep learning baseline (CNN/MLP)
├── federated_learning_fhs.py     # Federated learning comparison experiments
├── fl_results/                   # Output CSVs produced by the FL script
│   ├── convergence_summary.csv
│   ├── convergence_curve.csv
│   ├── privacy_utility.csv
│   ├── client_distribution.csv
│   └── noniid_sensitivity.csv
└── results/                      # Output of the deep learning script
```

---

## Federated Learning Experiments (`federated_learning_fhs.py`)

### What it does

Compares five federated learning algorithms for fetal heart-rate regression on
the IISc_FHS_DB dataset:

| Method | Colour (plots) | Description |
|--------|---------------|-------------|
| **FedAvg** | grey | McMahan et al. (2017) — vanilla parameter averaging |
| **FedProx** | blue | Li et al. (2020) — adds a proximal term to local loss |
| **SCAFFOLD** | orange | Karimireddy et al. (2020) — control variates to reduce client drift |
| **FedNova** | purple | Wang et al. (2020) — normalised gradient aggregation |
| **FedCrit-HEA** | **red (bold)** | *Proposed* — Heart Energy Adaptive aggregation |

The experiment pipeline:

1. **Feature extraction** — MFCC (13 coefficients, mean + std → 26-D vector)
   extracted from `.wav` files via *librosa*; synthetic placeholder features
   (seeded by subject ID) are used for subjects whose WAV stub cannot be
   decoded (see [Note on audio files](#note-on-audio-files) below).
2. **FL simulation** — 10 clients, 100 communication rounds, 5 independent
   seeds, IID data partitioning.
3. **Privacy–utility trade-off** — FedAvg with Laplace differential-privacy
   noise at six epsilon values `[0.5, 1, 2, 3, 5, 8]`.
4. **Non-IID sensitivity** — Dirichlet-based heterogeneous partitioning at
   `alpha ∈ {0.1, 0.3, 0.5, 1.0, ∞}`.

### Output CSVs (written to `fl_results/`)

| File | Contents |
|------|----------|
| `convergence_summary.csv` | Per-method: Round@90%, Conv. Round, MAE@50, Final MAE |
| `convergence_curve.csv` | Round-level MAE per method × seed |
| `privacy_utility.csv` | Mean ± std MAE vs. DP epsilon |
| `client_distribution.csv` | Final MAE per client × method × seed |
| `noniid_sensitivity.csv` | Mean ± std MAE per Dirichlet alpha × method |

All MAE values are in **bpm** (fetal heart rate beats per minute).

### How to run

#### 1. Install dependencies

```bash
pip install numpy pandas librosa scipy
```

> **Minimum versions tested:** numpy 1.24, pandas 2.0, librosa 0.10, scipy 1.10

#### 2. Run the experiment

```bash
python federated_learning_fhs.py
```

The script runs in about **2–3 minutes** on a standard laptop (CPU only).
Progress is printed to stdout and logged via Python's `logging` module.

#### 3. Check results

```bash
ls fl_results/
# convergence_summary.csv  convergence_curve.csv  privacy_utility.csv
# client_distribution.csv  noniid_sensitivity.csv
```

The convergence summary table is also printed to the terminal at the end.

---

## Note on audio files

Most `.wav` files in `dataset/` are 2-byte stubs (empty placeholders committed
to Git). Only a handful of subjects (`17`, `21`, `23`, `56`, `58`) contain
actual audio. The script automatically:

* Extracts real MFCC features from any readable WAV file.
* Falls back to a **deterministic synthetic feature vector** (seeded by subject
  ID) for subjects with unreadable stubs, so the pipeline runs end-to-end on
  the current repository state.

Once the full audio files are available, simply replace the `.wav` stubs with
the real recordings and rerun the script — no code changes are needed.

---

## Deep Learning Baseline (`deep_learning_fhs.py`)

A standalone 5-fold cross-validation experiment using a fully-connected neural
network (TensorFlow/Keras). Requires `IISc_features_complete_subset.csv`
(pre-extracted features, not included in this repo).

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python deep_learning_fhs.py
```

---

## Dataset citation

If you use this dataset, please cite the original IISc FHS dataset publication.

---

## Logging structures reference

The FL script uses the following data structures (as specified in the
experimental design):

```python
# history[method][seed]['round_mae'] = [mae_per_round]
history = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# client_history[method][seed][client_id] = final MAE (bpm)
client_history = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# privacy_results[epsilon] = list of MAE values across seeds
privacy_results = defaultdict(list)

# noniid_results[alpha][method] = list of MAE values across seeds
noniid_results = defaultdict(lambda: defaultdict(list))
```
