"""
FL vs CL Experimental Pipeline for Fetal Heart Rate Prediction
===============================================================
Extracts acoustic features from real .wav recordings in the IISc Fetal Heart
Sound dataset, then trains and compares a Federated Learning (FedAvg) model
against a Centralized Learning model.

Feature extraction
------------------
Each WAV recording is segmented into 10-second windows (50 % overlap).
Per window the following statistics (mean & std) are computed using librosa:
  • 20 MFCCs           → 40 features
  • 12 Chroma bands    → 24 features
  • Spectral centroid  →  2 features
  • Spectral roll-off  →  2 features
  • Spectral bandwidth →  2 features
  • Zero-crossing rate →  2 features
  • RMS energy         →  2 features
  Total: 74 features per window

Dataset
-------
Only subjects whose WAV file is non-empty AND whose FHR label appears in
Records.csv are used.  Subjects are each assigned to one FL client (natural
non-IID federated setting: each client holds data from one patient).

Metrics
-------
  Regression:     MAE, RMSE
  Classification: Precision, Recall  (label = FHR ≥ dataset median)
  Custom:         PPA  (% predictions within ±5 bpm of ground truth)

Visualizations (saved to results/fl_vs_cl/plots/)
--------------------------------------------------
  01_convergence_curve.png       – MAE/RMSE vs rounds/epochs ± SD shading
  02_performance_comparison.png  – Bar chart FL vs CL for all five metrics
  03_precision_recall_curve.png  – PR curves with AUPRC
  04_error_distribution.png      – Histogram + boxplot of prediction errors
  05_client_contribution.png     – Per-client MAE & PPA (FL only)
"""

import os
import json
import logging
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score,
    precision_recall_curve, auc,
)

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RECORDS_CSV = os.path.join(DATASET_DIR, "Records.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "fl_vs_cl")

for _sub in ["plots", "metrics"]:
    os.makedirs(os.path.join(RESULTS_DIR, _sub), exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW_SEC   = 10        # seconds per analysis window
STEP_SEC     = 5         # step between windows (50 % overlap)
N_MFCC       = 20        # number of MFCC coefficients

N_ROUNDS     = 50        # FL communication rounds
LOCAL_EPOCHS = 5         # local training epochs per FL round
CL_EPOCHS    = 250       # CL epochs  (= N_ROUNDS × LOCAL_EPOCHS)
BATCH_SIZE   = 16
LR           = 5e-4      # learning rate
GRAD_CLIP    = 5.0       # gradient clipping threshold
PPA_TOL      = 5.0       # ±5 bpm acceptable window for PPA
HR_MIN       = 100.0
HR_MAX       = 200.0
SEEDS        = [42, 43, 44]   # 3 seeds for statistical reliability

# Colour scheme: Blue = FL, Red = CL
FL_COLOR = "#2166AC"
CL_COLOR = "#D6604D"
FL_LIGHT = "#92C5DE"
CL_LIGHT = "#F4A582"

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – WAV Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_window_features(w: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 74 acoustic features from a single time window.
    Uses a pre-shared STFT magnitude spectrum to minimise recomputation.
    """
    S = np.abs(librosa.stft(w))
    S_db = librosa.power_to_db(S ** 2)

    mfcc   = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    sc     = librosa.feature.spectral_centroid(S=S, sr=sr)
    sro    = librosa.feature.spectral_rolloff(S=S, sr=sr)
    sbw    = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    zcr    = librosa.feature.zero_crossing_rate(w)
    rms    = librosa.feature.rms(S=S)

    return np.concatenate([
        mfcc.mean(1),   mfcc.std(1),    # 40
        chroma.mean(1), chroma.std(1),  # 24
        sc.mean(1),     sc.std(1),      #  2
        sro.mean(1),    sro.std(1),     #  2
        sbw.mean(1),    sbw.std(1),     #  2
        zcr.mean(1),    zcr.std(1),     #  2
        rms.mean(1),    rms.std(1),     #  2
    ]).astype(np.float32)               # total: 74


def extract_subject_features(wav_path: str) -> np.ndarray:
    """
    Load a WAV file and return a (n_windows × 74) feature matrix.
    Raises RuntimeError if the file has no usable audio content.
    """
    y, sr = librosa.load(wav_path, sr=None)
    if len(y) == 0:
        raise RuntimeError(f"Empty audio: {wav_path}")

    win_samples  = int(WINDOW_SEC * sr)
    step_samples = int(STEP_SEC   * sr)
    if len(y) < win_samples:
        raise RuntimeError(f"Audio too short ({len(y)/sr:.1f}s < {WINDOW_SEC}s): {wav_path}")

    windows = [
        y[s: s + win_samples]
        for s in range(0, len(y) - win_samples + 1, step_samples)
    ]
    return np.vstack([_extract_window_features(w, sr) for w in windows])


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> tuple:
    """
    Parse Records.csv, find subjects with a valid FHR label AND a non-empty
    WAV file, extract acoustic features, and return:
        X          – float32 array (total_windows × n_features)
        y          – float32 array (total_windows,)  FHR labels
        subject_ids– int array    (total_windows,)   subject index per window
        subjects   – list of subject numbers actually used
    """
    # ── Parse Records.csv ────────────────────────────────────────────────────
    hr_map = {}
    df_rec = pd.read_csv(RECORDS_CSV)
    for _, row in df_rec.iterrows():
        try:
            subj_id = int(row.iloc[0])
            hr_raw  = str(row.iloc[-1]).strip()
            if hr_raw and hr_raw not in ("-", "nan"):
                hr_map[subj_id] = float(hr_raw)
        except (ValueError, TypeError):
            pass

    # ── Match WAV files ───────────────────────────────────────────────────────
    MIN_FILE_BYTES = 100   # skip placeholder 2-byte files
    valid = []
    for subj_id, hr in sorted(hr_map.items()):
        wav_path = os.path.join(DATASET_DIR, f"subject_{subj_id:02d}.wav")
        if not os.path.exists(wav_path):
            continue
        if os.path.getsize(wav_path) < MIN_FILE_BYTES:
            continue
        valid.append((subj_id, hr, wav_path))

    if len(valid) == 0:
        raise RuntimeError(
            "No valid WAV files found in the dataset directory. "
            "Please ensure subject_XX.wav recordings are present."
        )

    log.info("Subjects with usable audio: %d  → %s",
             len(valid), [v[0] for v in valid])

    # ── Extract features ──────────────────────────────────────────────────────
    all_X, all_y, all_sid = [], [], []
    for idx, (subj_id, hr, wav_path) in enumerate(valid):
        log.info("  Extracting features: subject %02d  (FHR=%g bpm)  …", subj_id, hr)
        X_subj = extract_subject_features(wav_path)
        all_X.append(X_subj)
        all_y.extend([hr] * len(X_subj))
        all_sid.extend([idx] * len(X_subj))
        log.info("    → %d windows, %d features each", len(X_subj), X_subj.shape[1])

    X   = np.vstack(all_X).astype(np.float32)
    y   = np.array(all_y, dtype=np.float32)
    sid = np.array(all_sid, dtype=int)

    log.info("Dataset: %d windows × %d features  |  FHR: %.0f–%.0f bpm",
             X.shape[0], X.shape[1], y.min(), y.max())
    return X, y, sid, [v[0] for v in valid]


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Non-IID Client Partitioning
# ─────────────────────────────────────────────────────────────────────────────

def partition_by_subject(X_train, y_train, sid_train) -> list:
    """
    Natural non-IID partition: one FL client per subject.
    Returns list of (X_client, y_client) tuples.
    """
    client_ids = sorted(np.unique(sid_train))
    clients = []
    for cid in client_ids:
        mask = sid_train == cid
        clients.append((X_train[mask], y_train[mask]))
    sizes = [len(c[1]) for c in clients]
    log.info("Non-IID client partition: %d clients, window counts: %s",
             len(clients), sizes)
    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – NumPy MLP with FedAvg Support
# ─────────────────────────────────────────────────────────────────────────────

class NumpyMLP:
    """
    Lightweight fully-connected network implemented in NumPy.
    Architecture: input → 64 → 32 → 1  (ReLU hidden, linear output).
    Supports weight get/set for FedAvg aggregation.
    """

    def __init__(self, input_dim: int, hidden: tuple = (64, 32),
                 lr: float = LR, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        dims = [input_dim] + list(hidden) + [1]
        self.weights: list = []
        self.biases:  list = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32)
            W *= np.float32(np.sqrt(2.0 / fan_in))
            self.weights.append(W)
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    # ── forward / backward ────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self._acts = [X]
        self._zs   = []
        x = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ W + b
            self._zs.append(z)
            x = np.maximum(0.0, z) if i < len(self.weights) - 1 else z
            self._acts.append(x)
        return x

    def _backward(self, y_batch: np.ndarray):
        n     = y_batch.shape[0]
        delta = (2.0 / n) * (self._acts[-1] - y_batch.reshape(-1, 1))
        for i in range(len(self.weights) - 1, -1, -1):
            gW = self._acts[i].T @ delta
            gb = delta.sum(axis=0)
            # gradient clipping
            gW = np.clip(gW, -GRAD_CLIP, GRAD_CLIP)
            gb = np.clip(gb, -GRAD_CLIP, GRAD_CLIP)
            self.weights[i] -= self.lr * gW
            self.biases[i]  -= self.lr * gb
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self._zs[i - 1] > 0)

    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = BATCH_SIZE, rng_seed: int = 0) -> float:
        """Train one epoch; return mean MSE."""
        rng = np.random.default_rng(rng_seed)
        idx = rng.permutation(len(X))
        losses = []
        for s in range(0, len(X), batch_size):
            bi  = idx[s: s + batch_size]
            out = self._forward(X[bi])
            losses.append(float(np.mean((out.ravel() - y[bi]) ** 2)))
            self._backward(y[bi])
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward(X)
        return np.clip(out.ravel(), HR_MIN, HR_MAX)

    # ── weight access for FedAvg ──────────────────────────────────────────────

    def get_weights(self) -> dict:
        return {
            "W": [w.copy() for w in self.weights],
            "b": [b.copy() for b in self.biases],
        }

    def set_weights(self, params: dict):
        self.weights = [w.copy() for w in params["W"]]
        self.biases  = [b.copy() for b in params["b"]]


def fedavg_aggregate(client_params: list, client_sizes: list) -> dict:
    """Weighted (by dataset size) average of client weight dicts — FedAvg."""
    total = float(sum(client_sizes))
    agg   = {
        "W": [np.zeros_like(w) for w in client_params[0]["W"]],
        "b": [np.zeros_like(b) for b in client_params[0]["b"]],
    }
    for params, n in zip(client_params, client_sizes):
        frac = n / total
        for i, w in enumerate(params["W"]):
            agg["W"][i] += frac * w
        for i, b in enumerate(params["b"]):
            agg["b"][i] += frac * b
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _binary_labels(y: np.ndarray, thr: float) -> np.ndarray:
    return (y >= thr).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    hr_thr: float) -> dict:
    """
    Compute MAE, RMSE (regression) + Precision, Recall (binary classification
    at hr_thr) + PPA (fraction within ±PPA_TOL bpm).
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ppa  = float(np.mean(np.abs(y_true - y_pred) <= PPA_TOL) * 100.0)

    yb_true = _binary_labels(y_true, hr_thr)
    yb_pred = _binary_labels(y_pred, hr_thr)

    if len(np.unique(yb_true)) > 1:
        prec = float(precision_score(yb_true, yb_pred, zero_division=0))
        rec  = float(recall_score(yb_true, yb_pred, zero_division=0))
    else:
        prec = rec = float("nan")

    return {"MAE": mae, "RMSE": rmse, "Precision": prec, "Recall": rec, "PPA": ppa}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Centralized Learning
# ─────────────────────────────────────────────────────────────────────────────

def train_centralized(X_tr, y_tr, X_te, y_te,
                      input_dim: int, hr_thr: float,
                      seed: int, n_epochs: int = CL_EPOCHS) -> dict:
    """
    Train the MLP on the full (centralized) training set.
    Evaluates on test set every 5 epochs; records history & final metrics.
    """
    np.random.seed(seed)
    model   = NumpyMLP(input_dim, seed=seed)
    history = []

    for ep in range(1, n_epochs + 1):
        model.train_epoch(X_tr, y_tr, rng_seed=seed + ep)
        if ep % 5 == 0 or ep == n_epochs:
            m = compute_metrics(y_te, model.predict(X_te), hr_thr)
            m["epoch"] = ep
            history.append(m)

    y_pred_f = model.predict(X_te)
    final    = compute_metrics(y_te, y_pred_f, hr_thr)
    final["errors"]    = (y_te - y_pred_f).tolist()
    final["y_true_bin"] = _binary_labels(y_te, hr_thr).tolist()
    # Normalised score for PR curve
    span = float(HR_MAX - HR_MIN) or 1.0
    final["y_score"]   = ((y_pred_f - HR_MIN) / span).tolist()

    return {"history": history, "final_metrics": final}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 – Federated Learning (FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def train_federated(clients, X_te, y_te,
                    input_dim: int, hr_thr: float,
                    seed: int,
                    n_rounds: int      = N_ROUNDS,
                    local_epochs: int  = LOCAL_EPOCHS) -> dict:
    """
    FedAvg training loop.
    Each round: distribute global weights → local training → weighted aggregate.
    Records per-round global-model metrics and per-client final metrics.
    """
    np.random.seed(seed)
    global_model = NumpyMLP(input_dim, seed=seed)
    history      = []

    for rnd in range(1, n_rounds + 1):
        g_params        = global_model.get_weights()
        client_params   = []
        client_sizes    = []

        for c_idx, (X_c, y_c) in enumerate(clients):
            local = NumpyMLP(input_dim, seed=seed + c_idx)
            local.set_weights(g_params)
            for le in range(local_epochs):
                local.train_epoch(X_c, y_c, rng_seed=seed + rnd * 100 + c_idx * 10 + le)
            client_params.append(local.get_weights())
            client_sizes.append(len(y_c))

        global_model.set_weights(fedavg_aggregate(client_params, client_sizes))

        if rnd % 5 == 0 or rnd == n_rounds:
            y_pred = global_model.predict(X_te)
            # Guard: if predict returns NaN (numerical instability), skip
            if np.any(np.isnan(y_pred)):
                log.warning("  Round %d: NaN in predictions – skipping metric log", rnd)
                continue
            m       = compute_metrics(y_te, y_pred, hr_thr)
            m["round"] = rnd
            history.append(m)

    y_pred_f = global_model.predict(X_te)
    final    = compute_metrics(y_te, y_pred_f, hr_thr)
    final["errors"]     = (y_te - y_pred_f).tolist()
    final["y_true_bin"] = _binary_labels(y_te, hr_thr).tolist()
    span = float(HR_MAX - HR_MIN) or 1.0
    final["y_score"]    = ((y_pred_f - HR_MIN) / span).tolist()

    # Per-client final performance using global weights
    g_final = global_model.get_weights()
    client_metrics = []
    for c_idx, (X_c, y_c) in enumerate(clients):
        local = NumpyMLP(input_dim, seed=seed + c_idx)
        local.set_weights(g_final)
        y_c_pred = local.predict(X_c)
        cm = compute_metrics(y_c, y_c_pred, hr_thr)
        cm["client"] = c_idx + 1
        client_metrics.append(cm)

    return {"history": history, "final_metrics": final,
            "client_metrics": client_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 – Multi-Seed Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(X: np.ndarray, y: np.ndarray, sid: np.ndarray,
                    seeds: list = SEEDS) -> dict:
    """
    Run FL and CL experiments over multiple seeds.
    Each seed shuffles the train/test split but uses the same extracted features.
    """
    input_dim = X.shape[1]
    hr_thr    = float(np.median(y))
    log.info("HR threshold (median): %.1f bpm", hr_thr)

    all_cl, all_fl = [], []

    for seed in seeds:
        log.info("══ Seed %d ══════════════════════════════════════", seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))

        n_test  = max(1, int(len(X) * 0.20))
        te_idx  = idx[:n_test]
        tr_idx  = idx[n_test:]

        X_tr, y_tr, sid_tr = X[tr_idx], y[tr_idx], sid[tr_idx]
        X_te, y_te         = X[te_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
        X_te   = scaler.transform(X_te).astype(np.float32)

        clients = partition_by_subject(X_tr, y_tr, sid_tr)

        log.info("  CL training …")
        cl = train_centralized(X_tr, y_tr, X_te, y_te, input_dim, hr_thr, seed)

        log.info("  FL training (FedAvg) …")
        fl = train_federated(clients, X_te, y_te, input_dim, hr_thr, seed)

        log.info(
            "  CL → MAE=%.3f  RMSE=%.3f  Prec=%.3f  Rec=%.3f  PPA=%.1f%%",
            cl["final_metrics"]["MAE"], cl["final_metrics"]["RMSE"],
            cl["final_metrics"]["Precision"], cl["final_metrics"]["Recall"],
            cl["final_metrics"]["PPA"],
        )
        log.info(
            "  FL → MAE=%.3f  RMSE=%.3f  Prec=%.3f  Rec=%.3f  PPA=%.1f%%",
            fl["final_metrics"]["MAE"], fl["final_metrics"]["RMSE"],
            fl["final_metrics"]["Precision"], fl["final_metrics"]["Recall"],
            fl["final_metrics"]["PPA"],
        )
        all_cl.append(cl)
        all_fl.append(fl)

    return {"cl": all_cl, "fl": all_fl}


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Aggregation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg_history(results: list, metric: str, x_key: str):
    """
    Align per-step metric histories across seeds by their common x values.
    Returns (x_arr, mean_arr, std_arr).
    """
    x_sets = [set(m[x_key] for m in r["history"]) for r in results]
    common = sorted(x_sets[0].intersection(*x_sets[1:]))
    means, stds = [], []
    for xv in common:
        vals = [next(m[metric] for m in r["history"] if m[x_key] == xv)
                for r in results]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return np.array(common), np.array(means), np.array(stds)


def _agg_final(results: list) -> dict:
    """Average final metrics across seeds (NaN-safe)."""
    keys = ["MAE", "RMSE", "Precision", "Recall", "PPA"]
    out  = {}
    for k in keys:
        vals = [r["final_metrics"][k] for r in results
                if not np.isnan(r["final_metrics"][k])]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Publication-Quality Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, filename: str):
    path = os.path.join(RESULTS_DIR, "plots", filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", path)


def plot_convergence(cl_results: list, fl_results: list):
    """Plot 1 – MAE & RMSE convergence curves with ±1 SD shading."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Convergence Curve: Federated vs Centralized Learning",
        fontsize=14, fontweight="bold",
    )

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        cl_x, cl_m, cl_s = _agg_history(cl_results, metric, "epoch")
        fl_x, fl_m, fl_s = _agg_history(fl_results, metric, "round")

        ax.plot(cl_x, cl_m, color=CL_COLOR, linewidth=2,
                label="Centralized Learning")
        ax.fill_between(cl_x, cl_m - cl_s, cl_m + cl_s,
                        color=CL_COLOR, alpha=0.22)

        ax.plot(fl_x, fl_m, color=FL_COLOR, linewidth=2, linestyle="--",
                label="Federated Learning (FedAvg)")
        ax.fill_between(fl_x, fl_m - fl_s, fl_m + fl_s,
                        color=FL_COLOR, alpha=0.22)

        ax.set_xlabel("Epoch / Communication Round", fontsize=11)
        ax.set_ylabel(f"{metric} (bpm)", fontsize=11)
        ax.set_title(f"{metric} Convergence", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)

    plt.tight_layout()
    _save_fig(fig, "01_convergence_curve.png")


def plot_bar_comparison(cl_agg: dict, fl_agg: dict):
    """Plot 2 – Side-by-side bar chart for all five metrics."""
    metrics = list(cl_agg.keys())
    x, w    = np.arange(len(metrics)), 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b_cl = ax.bar(x - w / 2,
                  [cl_agg[m]["mean"] for m in metrics], w,
                  yerr=[cl_agg[m]["std"]  for m in metrics],
                  label="Centralized Learning", color=CL_COLOR,
                  alpha=0.88, capsize=5,
                  error_kw=dict(elinewidth=1.5))
    b_fl = ax.bar(x + w / 2,
                  [fl_agg[m]["mean"] for m in metrics], w,
                  yerr=[fl_agg[m]["std"]  for m in metrics],
                  label="Federated Learning", color=FL_COLOR,
                  alpha=0.88, capsize=5,
                  error_kw=dict(elinewidth=1.5))

    for bars in (b_cl, b_fl):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max(h * 0.02, 0.01),
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title(
        "Performance Comparison: Federated vs Centralized Learning",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save_fig(fig, "02_performance_comparison.png")


def plot_pr_curve(cl_results: list, fl_results: list):
    """Plot 3 – Precision-Recall curves with AUPRC annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for results, color, label in [
        (cl_results, CL_COLOR, "Centralized Learning"),
        (fl_results, FL_COLOR, "Federated Learning"),
    ]:
        for i, r in enumerate(results):
            fm     = r["final_metrics"]
            y_true = np.array(fm["y_true_bin"])
            y_sc   = np.array(fm["y_score"])
            if len(np.unique(y_true)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_sc)
            area = auc(rec, prec)
            is_last = i == len(results) - 1
            ax.plot(rec, prec, color=color,
                    linewidth=2 if is_last else 1,
                    alpha=1.0 if is_last else 0.4,
                    label=f"{label} (AUPRC={area:.3f})" if is_last else None)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve: Federated vs Centralized Learning",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    _save_fig(fig, "03_precision_recall_curve.png")


def plot_error_distribution(cl_results: list, fl_results: list):
    """Plot 4 – Error histogram and boxplot (all seeds pooled)."""
    cl_err = np.concatenate([r["final_metrics"]["errors"] for r in cl_results])
    fl_err = np.concatenate([r["final_metrics"]["errors"] for r in fl_results])

    lo  = min(cl_err.min(), fl_err.min())
    hi  = max(cl_err.max(), fl_err.max())
    bins = np.linspace(lo, hi, 25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Prediction Error Distribution: Federated vs Centralized Learning",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    ax.hist(cl_err, bins=bins, color=CL_COLOR, alpha=0.65,
            label="Centralized Learning", edgecolor="white")
    ax.hist(fl_err, bins=bins, color=FL_COLOR, alpha=0.65,
            label="Federated Learning", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (bpm)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Histogram", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    ax = axes[1]
    bp = ax.boxplot(
        [cl_err, fl_err],
        patch_artist=True, notch=False, widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, c in zip(bp["boxes"], [CL_COLOR, FL_COLOR]):
        patch.set_facecolor(c)
        patch.set_alpha(0.80)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Centralized\nLearning", "Federated\nLearning"], fontsize=11)
    ax.set_ylabel("Prediction Error (bpm)", fontsize=11)
    ax.set_title("Error Box-Plot", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--")

    plt.tight_layout()
    _save_fig(fig, "04_error_distribution.png")


def plot_client_contribution(fl_results: list, n_clients: int):
    """Plot 5 – Per-client MAE and PPA across FL clients (averaged over seeds)."""
    client_ids = list(range(1, n_clients + 1))
    palette    = ["#084594", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0"][:n_clients]

    mae_stats = []
    ppa_stats = []
    for c_idx in range(n_clients):
        maes = [r["client_metrics"][c_idx]["MAE"] for r in fl_results]
        ppas = [r["client_metrics"][c_idx]["PPA"] for r in fl_results]
        mae_stats.append((float(np.mean(maes)), float(np.std(maes))))
        ppa_stats.append((float(np.mean(ppas)), float(np.std(ppas))))

    x = np.arange(n_clients)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Client-wise Contribution (Federated Learning)",
                 fontsize=13, fontweight="bold")

    for ax, stats, ylabel, title in [
        (axes[0], mae_stats, "MAE (bpm)",  "MAE per Client"),
        (axes[1], ppa_stats, "PPA (%)",    "PPA per Client"),
    ]:
        means = [s[0] for s in stats]
        stds  = [s[1] for s in stats]
        bars  = ax.bar(x, means, width=0.6, yerr=stds,
                       color=palette, alpha=0.85, capsize=5,
                       error_kw=dict(elinewidth=1.5))
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(bar.get_height() * 0.02, 0.01),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Client {i}" for i in client_ids], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save_fig(fig, "05_client_contribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(cl_agg: dict, fl_agg: dict):
    json_path = os.path.join(RESULTS_DIR, "metrics", "fl_vs_cl_summary.json")
    with open(json_path, "w") as f:
        json.dump({"CL": cl_agg, "FL": fl_agg}, f, indent=2)
    log.info("Summary JSON → %s", json_path)

    txt_path = os.path.join(RESULTS_DIR, "fl_vs_cl_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  FL vs CL Comparison – Fetal Heart Rate Prediction\n")
        f.write("  IISc Fetal Heart Sound Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  {'Metric':<14} {'CL mean':>10} {'CL std':>8}  "
                f"{'FL mean':>10} {'FL std':>8}\n")
        f.write("  " + "─" * 54 + "\n")
        for m in cl_agg:
            f.write(f"  {m:<14} {cl_agg[m]['mean']:>10.4f} "
                    f"{cl_agg[m]['std']:>8.4f}  "
                    f"{fl_agg[m]['mean']:>10.4f} "
                    f"{fl_agg[m]['std']:>8.4f}\n")
        f.write("=" * 70 + "\n")
    log.info("Text report → %s", txt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("═" * 60)
    log.info("  FL vs CL Pipeline – IISc Fetal Heart Sound Dataset")
    log.info("═" * 60)
    log.info("Config: clients=dynamic  rounds=%d  local_epochs=%d  "
             "cl_epochs=%d  seeds=%s",
             N_ROUNDS, LOCAL_EPOCHS, CL_EPOCHS, SEEDS)

    # ── 1. Load & extract features ────────────────────────────────────────────
    X, y, sid, subjects = load_dataset()
    n_clients = len(subjects)   # one client per subject

    log.info("Using %d subjects as FL clients: %s", n_clients, subjects)

    # ── 2. Run experiments ────────────────────────────────────────────────────
    results    = run_experiments(X, y, sid, SEEDS)
    cl_results = results["cl"]
    fl_results = results["fl"]

    # ── 3. Aggregate metrics ──────────────────────────────────────────────────
    cl_agg = _agg_final(cl_results)
    fl_agg = _agg_final(fl_results)

    log.info("═" * 60)
    log.info("RESULTS  (mean ± std across %d seeds)", len(SEEDS))
    log.info("  %-14s  %10s  %10s", "Metric", "CL", "FL")
    log.info("  " + "─" * 38)
    for m in cl_agg:
        log.info("  %-14s  %10.4f  %10.4f", m,
                 cl_agg[m]["mean"], fl_agg[m]["mean"])

    # ── 4. Save summary ───────────────────────────────────────────────────────
    save_summary(cl_agg, fl_agg)

    # ── 5. Generate plots ─────────────────────────────────────────────────────
    log.info("Generating visualizations …")
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_convergence(cl_results, fl_results)
    plot_bar_comparison(cl_agg, fl_agg)
    plot_pr_curve(cl_results, fl_results)
    plot_error_distribution(cl_results, fl_results)
    plot_client_contribution(fl_results, n_clients)

    log.info("═" * 60)
    log.info("Pipeline complete. All outputs in: %s", RESULTS_DIR)
    log.info("═" * 60)
