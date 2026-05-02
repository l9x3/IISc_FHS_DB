#!/usr/bin/env python3
"""
Federated Learning Comparison for Fetal Heart Sound Dataset (IISc_FHS_DB)
==========================================================================
Compares five FL methods on fetal heart-rate (FHR) regression:
  - FedAvg    (grey)
  - FedProx   (blue)
  - SCAFFOLD  (orange)
  - FedNova   (purple)
  - FedCrit-HEA (proposed, red, bold)

Data:
  - Audio features extracted from dataset/*.wav via MFCC
  - Labels: FHR (bpm) from dataset/Records.csv

Outputs (written to fl_results/):
  convergence_summary.csv   – per-method convergence metrics across seeds
  convergence_curve.csv     – round-level MAE per method/seed
  privacy_utility.csv       – differential-privacy epsilon vs. MAE
  client_distribution.csv   – per-client final MAE per method/seed
  noniid_sensitivity.csv    – Dirichlet alpha vs. MAE per method

Usage:
  python federated_learning_fhs.py

Requirements:
  pip install numpy pandas librosa scipy
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RECORDS_CSV = os.path.join(DATASET_DIR, "Records.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "fl_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment hyper-parameters
# ---------------------------------------------------------------------------
NUM_ROUNDS = 100
METHODS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedCrit-HEA"]
SEEDS = [0, 1, 2, 3, 4]
NUM_CLIENTS = 10
CLIENTS = list(range(1, NUM_CLIENTS + 1))

LOCAL_STEPS = 5          # local SGD steps per FL round
LR = 0.01                # local learning rate
MU_FEDPROX = 0.01        # FedProx proximal coefficient

N_MFCC = 13              # number of MFCC coefficients
AUDIO_SR = 4000          # target sample rate for loading .wav files

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _try_import_librosa():
    try:
        import librosa as _lib
        return _lib
    except ImportError:
        return None


def extract_mfcc(wav_path: str, n_mfcc: int = N_MFCC, sr: int = AUDIO_SR) -> np.ndarray:
    """
    Extract MFCC features from a WAV file.

    Strategy (in order):
    1. Try librosa (handles most standard WAV formats).
    2. Try scipy.io.wavfile as a fallback.
    3. If both fail (e.g. placeholder / corrupt file), return None so the
       caller can substitute synthetic features.

    Returns a 1-D array of length 2*n_mfcc (mean + std of each coefficient),
    or None if the file cannot be decoded.
    """
    # ── Attempt 1: librosa ────────────────────────────────────────────────────
    librosa = _try_import_librosa()
    if librosa is not None:
        try:
            y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
            feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            return feats.astype(np.float32)
        except Exception:
            pass  # fall through to scipy

    # ── Attempt 2: scipy + hand-crafted MFCC approximation ───────────────────
    try:
        from scipy.io import wavfile
        from scipy.fft import dct
        import scipy.signal as spsig

        file_sr, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        # Resample to target sr via decimation/interpolation
        if file_sr != sr:
            num = int(len(data) * sr / file_sr)
            data = spsig.resample(data, num).astype(np.float32)
        # Normalise
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak

        # Compute simple MFCCs via FFT + mel filterbank
        frame_len = int(0.025 * sr)  # 25 ms
        hop_len = int(0.010 * sr)    # 10 ms
        n_fft = 512
        frames = np.lib.stride_tricks.sliding_window_view(data, frame_len)[::hop_len]
        windowed = frames * np.hanning(frame_len)
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

        # Mel filterbank (triangular filters)
        fmin, fmax = 0, sr // 2
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_pts = np.linspace(mel_min, mel_max, n_mfcc + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mfcc, n_freqs), dtype=np.float32)
        for m in range(1, n_mfcc + 1):
            lo, cen, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
            for k in range(lo, cen):
                if cen != lo:
                    filterbank[m - 1, k] = (k - lo) / (cen - lo)
            for k in range(cen, hi):
                if hi != cen:
                    filterbank[m - 1, k] = (hi - k) / (hi - cen)

        mel_energy = np.log(spectrum @ filterbank.T + 1e-10)  # (frames, n_mfcc)
        mfcc_vals = dct(mel_energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]

        feats = np.concatenate([mfcc_vals.mean(axis=0), mfcc_vals.std(axis=0)])
        return feats.astype(np.float32)
    except Exception:
        pass

    # ── Placeholder ───────────────────────────────────────────────────────────
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    """
    Load feature vectors and FHR labels for all subjects that have both a
    .wav file and a numeric FHR entry in Records.csv.

    Returns
    -------
    X : np.ndarray, shape (N, 2*N_MFCC)
    y : np.ndarray, shape (N,)
    subjects : list[int]
    """
    records = pd.read_csv(RECORDS_CSV)
    records.columns = [c.strip() for c in records.columns]

    # Identify the FHR column (the last column contains bpm values)
    fhr_col = records.columns[-1]

    X_list, y_list, subject_list = [], [], []

    for _, row in records.iterrows():
        try:
            subj = int(row["Subject"])
        except (ValueError, KeyError):
            continue

        # Parse FHR label – skip if missing or non-numeric
        fhr_raw = str(row[fhr_col]).strip()
        if fhr_raw in ("", "-", "nan"):
            continue
        try:
            fhr_val = float(fhr_raw)
        except ValueError:
            continue

        # Locate the corresponding .wav file (zero-padded or plain)
        wav_path = os.path.join(DATASET_DIR, f"subject_{subj:02d}.wav")
        if not os.path.exists(wav_path):
            wav_path = os.path.join(DATASET_DIR, f"subject_{subj}.wav")
        if not os.path.exists(wav_path):
            continue

        feats = extract_mfcc(wav_path)
        if feats is None:
            # PLACEHOLDER: the WAV file is an empty/corrupt stub in this
            # repository checkout. A deterministic synthetic feature vector
            # seeded by subject ID is used so that clients remain
            # distinguishable during simulation. Replace with real MFCC
            # extraction once the full audio files are available.
            log.warning(
                "subject_%02d: audio unreadable – using synthetic placeholder features",
                subj,
            )
            rng_s = np.random.default_rng(subj)
            feats = rng_s.standard_normal(2 * N_MFCC).astype(np.float32)
        X_list.append(feats)
        y_list.append(fhr_val)
        subject_list.append(subj)

    if len(X_list) == 0:
        raise RuntimeError(
            "No samples loaded. Verify that dataset/ contains .wav files and "
            "Records.csv has matching subject IDs with numeric FHR values."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Standardise features globally (simple baseline; in real FL this would
    # be done per-client or via a private mean/std estimate)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Standardise target to zero mean / unit std so SGD gradients remain at
    # a reasonable scale. The load_dataset caller receives the raw bpm values
    # alongside the scale factors so that MAE can be reported in original units.
    y_mean = float(y.mean())
    y_std = float(y.std()) + 1e-8
    y_norm = (y - y_mean) / y_std

    log.info("Dataset: %d samples, %d features, FHR range %.0f–%.0f bpm",
             len(y), X.shape[1], y.min(), y.max())
    return X, y_norm, subject_list, y_mean, y_std


# ---------------------------------------------------------------------------
# Client partitioning
# ---------------------------------------------------------------------------

def partition_iid(X, y, num_clients: int, seed: int):
    """Randomly shuffle and split data into num_clients equal partitions."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, num_clients)
    return [(X[s], y[s]) for s in splits]


def partition_noniid_dirichlet(X, y, num_clients: int, alpha: float, seed: int):
    """
    Partition data using a Dirichlet distribution with concentration *alpha*.

    Smaller alpha → more heterogeneous (non-IID) partitions.
    alpha = inf  → IID (uniform) partition.
    """
    if alpha == float("inf"):
        return partition_iid(X, y, num_clients, seed)

    rng = np.random.default_rng(seed)
    n = len(y)
    # Assign each sample to a client using Dirichlet-drawn proportions
    proportions = rng.dirichlet(alpha * np.ones(num_clients))
    cumulative = np.cumsum(proportions)
    idx = rng.permutation(n)
    splits = []
    prev = 0
    for i, c in enumerate(cumulative):
        end = int(c * n) if i < num_clients - 1 else n
        splits.append(idx[prev:end])
        prev = end
    return [(X[s], y[s]) for s in splits]


# ---------------------------------------------------------------------------
# Local linear regression model (weights + bias)
# ---------------------------------------------------------------------------

class LinearModel:
    """Simple linear regression model represented as (weights, bias)."""

    def __init__(self, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, (n_features,)).astype(np.float64)
        self.b = 0.0

    @property
    def params(self):
        return np.concatenate([self.w, [self.b]])

    @params.setter
    def params(self, p):
        self.w = p[:-1].copy()
        self.b = float(p[-1])

    def predict(self, X):
        return X @ self.w + self.b

    def loss(self, X, y):
        """MSE loss."""
        return float(np.mean((self.predict(X) - y) ** 2))

    def gradient(self, X, y):
        """MSE gradient w.r.t. (w, b)."""
        n = len(y)
        err = self.predict(X) - y
        dw = 2 * (X.T @ err) / n
        db = 2 * err.mean()
        return np.concatenate([dw, [db]])

    def mae(self, X, y):
        return float(np.mean(np.abs(self.predict(X) - y)))


# ---------------------------------------------------------------------------
# Differential-privacy noise helper
# ---------------------------------------------------------------------------

def dp_noise(params: np.ndarray, epsilon: float, sensitivity: float = 1.0,
             rng: np.random.Generator = None) -> np.ndarray:
    """Add Laplace noise calibrated to (epsilon, sensitivity) DP guarantee."""
    if rng is None:
        rng = np.random.default_rng()
    scale = sensitivity / epsilon
    return params + rng.laplace(0, scale, size=params.shape)


# ---------------------------------------------------------------------------
# FL algorithm implementations
# ---------------------------------------------------------------------------

def _sgd_step(model: LinearModel, X, y, lr: float,
              global_params=None, mu: float = 0.0,
              control_variate=None, global_control=None):
    """
    One SGD step with optional FedProx proximal term and SCAFFOLD correction.

    Parameters
    ----------
    mu : float
        FedProx proximal coefficient (0 → standard SGD).
    control_variate : array or None
        SCAFFOLD client control variate c_i.
    global_control : array or None
        SCAFFOLD server control variate c.
    """
    grad = model.gradient(X, y)

    # FedProx proximal gradient
    if global_params is not None and mu > 0:
        grad += mu * (model.params - global_params)

    # SCAFFOLD correction
    if control_variate is not None and global_control is not None:
        grad += (-control_variate + global_control)

    # Gradient clipping (guards against very small / skewed non-IID partitions)
    norm = np.linalg.norm(grad)
    if norm > 5.0:
        grad = grad * (5.0 / norm)

    model.params = model.params - lr * grad


def run_fedavg(client_data, n_features, num_rounds, local_steps, lr, seed):
    """Standard FedAvg: average client parameters each round."""
    rng = np.random.default_rng(seed)
    global_model = LinearModel(n_features, seed=seed)
    round_mae = []

    for r in range(num_rounds):
        local_params = []
        for X_c, y_c in client_data:
            if len(y_c) == 0:
                continue
            local = LinearModel(n_features)
            local.params = global_model.params.copy()
            for _ in range(local_steps):
                _sgd_step(local, X_c, y_c, lr)
            local_params.append(local.params)

        global_model.params = np.mean(local_params, axis=0)
        # Evaluate globally
        all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
        all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
        round_mae.append(global_model.mae(all_X, all_y))

    return global_model, round_mae


def run_fedprox(client_data, n_features, num_rounds, local_steps, lr, mu, seed):
    """FedProx: adds proximal regularisation ||w - w_global||^2 in local loss."""
    global_model = LinearModel(n_features, seed=seed)
    round_mae = []

    for r in range(num_rounds):
        global_params = global_model.params.copy()
        local_params = []
        for X_c, y_c in client_data:
            if len(y_c) == 0:
                continue
            local = LinearModel(n_features)
            local.params = global_params.copy()
            for _ in range(local_steps):
                _sgd_step(local, X_c, y_c, lr, global_params=global_params, mu=mu)
            local_params.append(local.params)

        global_model.params = np.mean(local_params, axis=0)
        all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
        all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
        round_mae.append(global_model.mae(all_X, all_y))

    return global_model, round_mae


def run_scaffold(client_data, n_features, num_rounds, local_steps, lr, seed):
    """
    SCAFFOLD: maintains client and server control variates to reduce
    client-drift in heterogeneous settings.
    """
    global_model = LinearModel(n_features, seed=seed)
    n_params = len(global_model.params)
    num_clients = len(client_data)

    # Initialise control variates at zero
    c_global = np.zeros(n_params)
    c_clients = [np.zeros(n_params) for _ in range(num_clients)]
    round_mae = []

    for r in range(num_rounds):
        global_params = global_model.params.copy()
        local_params_list = []
        new_c_clients = []

        for i, (X_c, y_c) in enumerate(client_data):
            if len(y_c) == 0:
                new_c_clients.append(c_clients[i])
                continue
            local = LinearModel(n_features)
            local.params = global_params.copy()
            for _ in range(local_steps):
                _sgd_step(local, X_c, y_c, lr,
                          control_variate=c_clients[i],
                          global_control=c_global)

            # Update client control variate (option II from SCAFFOLD paper)
            delta = global_params - local.params
            new_c_i = c_clients[i] - c_global + delta / (local_steps * lr)
            new_c_clients.append(new_c_i)
            local_params_list.append(local.params)

        if local_params_list:
            global_model.params = np.mean(local_params_list, axis=0)
            # Update global control variate
            active = [nc for nc, (_, y_c) in zip(new_c_clients, client_data)
                      if len(y_c) > 0]
            c_global = np.mean(active, axis=0)
        c_clients = new_c_clients

        all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
        all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
        round_mae.append(global_model.mae(all_X, all_y))

    return global_model, round_mae


def run_fednova(client_data, n_features, num_rounds, local_steps, lr, seed):
    """
    FedNova: normalises client updates by the number of local steps taken to
    correct for objective inconsistency.
    """
    global_model = LinearModel(n_features, seed=seed)
    round_mae = []

    for r in range(num_rounds):
        global_params = global_model.params.copy()
        normalized_updates = []

        for X_c, y_c in client_data:
            if len(y_c) == 0:
                continue
            local = LinearModel(n_features)
            local.params = global_params.copy()
            for _ in range(local_steps):
                _sgd_step(local, X_c, y_c, lr)
            # Normalise by effective number of local steps
            update = (global_params - local.params) / local_steps
            normalized_updates.append(update)

        if normalized_updates:
            avg_update = np.mean(normalized_updates, axis=0)
            global_model.params = global_params - avg_update * local_steps

        all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
        all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
        round_mae.append(global_model.mae(all_X, all_y))

    return global_model, round_mae


def _heart_energy(X, y, model: LinearModel) -> float:
    """
    Heart Energy score for FedCrit-HEA aggregation.

    Combines prediction quality (low MAE → high energy) and label variance
    (higher physiological variability → more informative client signal).
    The score is clipped to prevent numerical explosion during early rounds.
    """
    mae = model.mae(X, y) + 1e-8
    var = float(np.var(y)) + 1e-8
    return float(np.clip(var / mae, 1e-6, 1e6))


def run_fedcrit_hea(client_data, n_features, num_rounds, local_steps, lr, seed):
    """
    FedCrit-HEA (proposed):
    Federated Critical Aggregation with Heart Energy Adaptation.

    Aggregation weights are proportional to each client's Heart Energy score –
    clients whose local model achieves better MAE relative to label variance
    receive higher weight in the global update.
    """
    global_model = LinearModel(n_features, seed=seed)
    round_mae = []

    for r in range(num_rounds):
        global_params = global_model.params.copy()
        local_models = []

        for X_c, y_c in client_data:
            if len(y_c) == 0:
                continue
            local = LinearModel(n_features)
            local.params = global_params.copy()
            for _ in range(local_steps):
                _sgd_step(local, X_c, y_c, lr)
            local_models.append((local, X_c, y_c))

        if local_models:
            # Compute Heart Energy weights with temperature-scaled softmax.
            # A uniform-mixing coefficient λ (20 %) provides a minimum weight
            # floor so that no single client can fully dominate aggregation.
            n_active = len(local_models)
            energies = np.array([_heart_energy(X_c, y_c, m)
                                 for m, X_c, y_c in local_models])
            log_energies = np.log(energies + 1e-10)
            log_energies -= log_energies.max()  # numerical stability
            exp_e = np.exp(log_energies)
            softmax_w = exp_e / (exp_e.sum() + 1e-12)
            # Blend: 80% energy-based + 20% uniform
            lam = 0.2
            weights = (1 - lam) * softmax_w + lam / n_active
            # Weighted aggregation
            global_model.params = sum(
                w * m.params for w, (m, _, __) in zip(weights, local_models)
            )

        all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
        all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
        round_mae.append(global_model.mae(all_X, all_y))

    return global_model, round_mae


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def run_method(method: str, client_data, n_features, num_rounds,
               local_steps, lr, mu, seed):
    """Run a named FL method and return (global_model, round_mae list)."""
    if method == "FedAvg":
        return run_fedavg(client_data, n_features, num_rounds, local_steps, lr, seed)
    elif method == "FedProx":
        return run_fedprox(client_data, n_features, num_rounds, local_steps, lr, mu, seed)
    elif method == "SCAFFOLD":
        return run_scaffold(client_data, n_features, num_rounds, local_steps, lr, seed)
    elif method == "FedNova":
        return run_fednova(client_data, n_features, num_rounds, local_steps, lr, seed)
    elif method == "FedCrit-HEA":
        return run_fedcrit_hea(client_data, n_features, num_rounds, local_steps, lr, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Storage dictionaries  (matching the spec)
# ---------------------------------------------------------------------------
history = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# history[method][seed]['round_mae'] = [mae_per_round]

client_history = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
# client_history[method][seed][client_id] = final MAE

privacy_results = defaultdict(list)
# privacy_results[epsilon] = list of MAE values

noniid_results = defaultdict(lambda: defaultdict(list))
# noniid_results[alpha][method] = list of MAE values


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    log.info("Loading dataset …")
    X, y_norm, subjects, y_mean, y_std = load_dataset()
    n_features = X.shape[1]

    def to_bpm(mae_norm):
        """Convert normalised MAE back to bpm units."""
        return mae_norm * y_std

    # ── 1. Main convergence + client-distribution experiment ─────────────────
    log.info("Running convergence experiments (%d methods × %d seeds × %d rounds) …",
             len(METHODS), len(SEEDS), NUM_ROUNDS)

    for method in METHODS:
        for seed in SEEDS:
            client_data = partition_iid(X, y_norm, NUM_CLIENTS, seed)

            # Single run: returns the trained global model AND round-level MAE
            global_model, round_mae_norm = run_method(
                method, client_data, n_features,
                NUM_ROUNDS, LOCAL_STEPS, LR, MU_FEDPROX, seed
            )
            round_mae = [to_bpm(m) for m in round_mae_norm]
            history[method][seed]["round_mae"] = round_mae

            # Per-client final MAE from the same global model
            for c_idx, (X_c, y_c) in enumerate(client_data):
                if len(y_c) == 0:
                    continue
                client_history[method][seed][CLIENTS[c_idx]] = to_bpm(
                    global_model.mae(X_c, y_c)
                )

            log.info("  %-12s  seed=%d  final_MAE=%.4f bpm",
                     method, seed, round_mae[-1])

    # ── 2. Privacy-utility trade-off ─────────────────────────────────────────
    log.info("Running privacy-utility experiments …")
    epsilons = [0.5, 1, 2, 3, 5, 8]

    for eps in epsilons:
        for seed in SEEDS:
            client_data = partition_iid(X, y_norm, NUM_CLIENTS, seed)
            global_model = LinearModel(n_features, seed=seed)

            for r in range(NUM_ROUNDS):
                global_params = global_model.params.copy()
                local_params = []
                rng = np.random.default_rng(seed + r)
                for X_c, y_c in client_data:
                    if len(y_c) == 0:
                        continue
                    local = LinearModel(n_features)
                    local.params = global_params.copy()
                    for _ in range(LOCAL_STEPS):
                        _sgd_step(local, X_c, y_c, LR)
                    # Apply DP noise to gradients before aggregation
                    noisy_params = dp_noise(local.params, eps,
                                           sensitivity=1.0, rng=rng)
                    local_params.append(noisy_params)
                if local_params:
                    global_model.params = np.mean(local_params, axis=0)

            all_X = np.vstack([d[0] for d in client_data if len(d[1]) > 0])
            all_y = np.concatenate([d[1] for d in client_data if len(d[1]) > 0])
            privacy_results[eps].append(to_bpm(global_model.mae(all_X, all_y)))
        log.info("  epsilon=%.1f  mean_MAE=%.4f bpm", eps,
                 float(np.mean(privacy_results[eps])))

    # ── 3. Non-IID sensitivity ────────────────────────────────────────────────
    log.info("Running non-IID sensitivity experiments …")
    alphas = [0.1, 0.3, 0.5, 1.0, float("inf")]

    for alpha in alphas:
        for method in METHODS:
            for seed in SEEDS:
                client_data = partition_noniid_dirichlet(X, y_norm, NUM_CLIENTS, alpha, seed)
                _, round_mae_a_norm = run_method(
                    method, client_data, n_features,
                    NUM_ROUNDS, LOCAL_STEPS, LR, MU_FEDPROX, seed
                )
                noniid_results[alpha][method].append(to_bpm(round_mae_a_norm[-1]))
        log.info("  alpha=%s  done", alpha)

    # ── Build and export DataFrames ───────────────────────────────────────────

    # 1. Convergence summary
    def compute_convergence_table(hist):
        rows = []
        for method in hist:
            all_seeds = []
            for seed in hist[method]:
                mae_curve = np.array(hist[method][seed]["round_mae"])
                final_mae = mae_curve[-1]
                target = final_mae * 1.10  # 90% convergence threshold
                reached = np.where(mae_curve <= target)[0]
                round_90 = int(reached[0]) if len(reached) > 0 else len(mae_curve)
                diffs = np.abs(np.diff(mae_curve))
                conv_idx = np.where(diffs < 1e-3)[0]
                conv_round = int(conv_idx[0]) if len(conv_idx) > 0 else len(mae_curve)
                mae_50 = float(mae_curve[49]) if len(mae_curve) > 49 else float(mae_curve[-1])
                all_seeds.append([round_90, conv_round, mae_50, final_mae])
            all_seeds = np.array(all_seeds)
            rows.append({
                "Method": method,
                "Round@90% (MAE↓)": float(np.mean(all_seeds[:, 0])),
                "Conv. Round": float(np.mean(all_seeds[:, 1])),
                "MAE@50": float(np.mean(all_seeds[:, 2])),
                "Final MAE": float(np.mean(all_seeds[:, 3])),
            })
        return pd.DataFrame(rows)

    convergence_table = compute_convergence_table(history)

    # 2. Convergence curve
    def build_convergence_curve_df(hist):
        data = []
        for method in hist:
            for seed in hist[method]:
                mae_curve = hist[method][seed]["round_mae"]
                for r, mae in enumerate(mae_curve):
                    data.append({"Method": method, "Seed": seed,
                                 "Round": r, "MAE": mae})
        return pd.DataFrame(data)

    convergence_curve_df = build_convergence_curve_df(history)

    # 3. Privacy utility
    privacy_df = pd.DataFrame([
        {
            "Epsilon": eps,
            "MAE": float(np.mean(privacy_results[eps])),
            "Std": float(np.std(privacy_results[eps])),
        }
        for eps in epsilons
    ])

    # 4. Client distribution
    def build_client_df(ch):
        data = []
        for method in ch:
            for seed in ch[method]:
                for client in ch[method][seed]:
                    data.append({
                        "Method": method,
                        "Seed": seed,
                        "Client": client,
                        "MAE": ch[method][seed][client],
                    })
        return pd.DataFrame(data)

    client_df = build_client_df(client_history)

    # 5. Non-IID sensitivity
    noniid_rows = []
    for alpha in noniid_results:
        for method in noniid_results[alpha]:
            values = noniid_results[alpha][method]
            noniid_rows.append({
                "Alpha": alpha,
                "Method": method,
                "MAE": float(np.mean(values)),
                "Std": float(np.std(values)),
            })
    noniid_df = pd.DataFrame(noniid_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    def out(fname):
        return os.path.join(RESULTS_DIR, fname)

    convergence_table.to_csv(out("convergence_summary.csv"), index=False)
    convergence_curve_df.to_csv(out("convergence_curve.csv"), index=False)
    privacy_df.to_csv(out("privacy_utility.csv"), index=False)
    client_df.to_csv(out("client_distribution.csv"), index=False)
    noniid_df.to_csv(out("noniid_sensitivity.csv"), index=False)

    log.info("Results written to %s/", RESULTS_DIR)
    log.info("  convergence_summary.csv")
    log.info("  convergence_curve.csv")
    log.info("  privacy_utility.csv")
    log.info("  client_distribution.csv")
    log.info("  noniid_sensitivity.csv")

    # ── Print convergence table to console ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Convergence Summary")
    print("=" * 60)
    print(convergence_table.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
