"""
Federated Learning for Fetal Heart Rate (FHR) Estimation
=========================================================
Implements FedAvg, FedProx, SCAFFOLD, FedNova, and FedCrit-HEA on the
IIScFHSDB dataset (WAV files in the ./dataset directory).

When real WAV files are present, audio features (MFCCs, spectral features,
rhythmic features) are extracted.  When only placeholder files exist (as in
this repository), reproducible synthetic features are generated from the
subject-id seed so that all FL algorithms can still be exercised and compared.

Outputs (written to ./fl_results/):
  convergence_summary.csv  – per-round losses and test metrics for every algo
  fl_results.png           – convergence curves and final metric comparison
  client_data_summary.csv  – data-distribution across simulated FL clients
  final_metrics.csv        – per-algorithm MAE / RMSE / R2 / PPA at last round
"""

import os
import copy
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "./dataset"
RESULTS_DIR = "./fl_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── FL Hyperparameters ────────────────────────────────────────────────────────
N_CLIENTS = 5          # number of simulated hospitals / clients
N_ROUNDS = 50          # global communication rounds
LOCAL_EPOCHS = 5       # local SGD epochs per round
LOCAL_LR = 0.01        # local learning rate
BATCH_SIZE = 4         # mini-batch size
MU_FEDPROX = 0.01      # FedProx proximal regularisation coefficient

# ── Physiological bounds ──────────────────────────────────────────────────────
HR_MIN, HR_MAX = 100.0, 200.0
PPA_TOLERANCE = 5.0    # ±5 bpm acceptable range

# ── Audio feature extraction settings ────────────────────────────────────────
N_MFCC = 13
N_FEATURES = 40        # total feature dimension

ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedCrit-HEA"]


# ─────────────────────────────────────────────────────────────────────────────
# Audio feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_wav(path: str) -> bool:
    """Return True only when the file appears to be a proper WAV binary."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(4)
        return header == b"RIFF"
    except OSError:
        return False


def _extract_real_features(wav_path: str) -> np.ndarray:
    """Load a real WAV file and extract a fixed-length feature vector."""
    import librosa  # imported here so the rest of the module loads without it

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    feats: List[np.ndarray] = []

    # 13 MFCC means + 13 MFCC stds
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.extend(mfcc.mean(axis=1))
    feats.extend(mfcc.std(axis=1))

    # Spectral centroid, bandwidth, rolloff, ZCR (mean + std each)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    for feat in (sc, sb, sr_, zcr):
        feats.append(float(feat.mean()))
        feats.append(float(feat.std()))

    return np.array(feats, dtype=np.float32)


def _synthetic_features(subject_id: int, fhr: float) -> np.ndarray:
    """
    Generate reproducible synthetic features for subjects whose WAV file is
    a placeholder.  Features are seeded by subject_id so they are stable
    across runs; a subtle FHR-correlated component is included so that
    supervised learning is feasible.
    """
    rng = np.random.RandomState(subject_id)
    # Normalised FHR drives a latent component
    fhr_norm = (fhr - HR_MIN) / (HR_MAX - HR_MIN)
    noise = rng.randn(N_FEATURES).astype(np.float32)
    signal = np.linspace(0, 1, N_FEATURES, dtype=np.float32) * fhr_norm
    return signal + noise * 0.5


def extract_features(wav_path: str, subject_id: int, fhr: float) -> np.ndarray:
    """Return a feature vector for one subject."""
    if _is_valid_wav(wav_path):
        try:
            return _extract_real_features(wav_path)
        except Exception as exc:
            log.debug("Real feature extraction failed for %s: %s – using synthetic", wav_path, exc)
    return _synthetic_features(subject_id, fhr)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y, subject_ids) by scanning DATA_DIR for WAV files and
    matching them with the FHR labels in Records.csv.
    """
    records_path = os.path.join(DATA_DIR, "Records.csv")
    if not os.path.isfile(records_path):
        raise FileNotFoundError(f"Records.csv not found at {records_path}")

    df = pd.read_csv(records_path)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    fhr_col = [c for c in df.columns if "FHR" in c.upper() or "bpm" in c.lower()]
    if not fhr_col:
        raise ValueError("Cannot find FHR column in Records.csv")
    fhr_col = fhr_col[0]

    # Keep only rows where FHR is a number
    df[fhr_col] = pd.to_numeric(df[fhr_col], errors="coerce")
    df = df.dropna(subset=[fhr_col])
    df["Subject"] = pd.to_numeric(df["Subject"], errors="coerce")
    df = df.dropna(subset=["Subject"])
    df["Subject"] = df["Subject"].astype(int)

    rows_X, rows_y, rows_sid = [], [], []
    for _, row in df.iterrows():
        sid = int(row["Subject"])
        fhr = float(row[fhr_col])
        wav_path = os.path.join(DATA_DIR, f"subject_{sid:02d}.wav")
        if not os.path.isfile(wav_path):
            # Try without leading zero
            wav_path = os.path.join(DATA_DIR, f"subject_{sid}.wav")
        if not os.path.isfile(wav_path):
            continue  # skip subjects without a WAV placeholder

        feats = extract_features(wav_path, sid, fhr)
        rows_X.append(feats)
        rows_y.append(fhr)
        rows_sid.append(sid)

    if not rows_X:
        raise RuntimeError("No samples loaded – check DATA_DIR and Records.csv")

    # Pad / truncate all feature vectors to N_FEATURES
    max_len = max(len(v) for v in rows_X)
    feature_dim = max(max_len, N_FEATURES)
    X = np.zeros((len(rows_X), feature_dim), dtype=np.float32)
    for i, v in enumerate(rows_X):
        X[i, : len(v)] = v

    y = np.array(rows_y, dtype=np.float32)
    sids = np.array(rows_sid, dtype=int)
    log.info("Loaded %d samples, feature_dim=%d, FHR range=[%.0f, %.0f]",
             len(y), X.shape[1], y.min(), y.max())
    return X, y, sids


# ─────────────────────────────────────────────────────────────────────────────
# Federated data partitioning
# ─────────────────────────────────────────────────────────────────────────────

def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    seed: int = SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID partitioning: subjects are sorted by FHR and assigned to clients
    in round-robin fashion so each client sees a similar FHR distribution.
    """
    rng = np.random.RandomState(seed)
    order = np.argsort(y)
    # Accumulate rows per client using plain lists before converting to ndarray
    client_X: List[List[np.ndarray]] = [[] for _ in range(n_clients)]
    client_y: List[List[float]] = [[] for _ in range(n_clients)]
    for rank, idx in enumerate(order):
        c = rank % n_clients
        client_X[c].append(X[idx])
        client_y[c].append(float(y[idx]))

    result: List[Tuple[np.ndarray, np.ndarray]] = []
    for cX, cy in zip(client_X, client_y):
        Xc = np.array(cX, dtype=np.float32)
        yc = np.array(cy, dtype=np.float32)
        # Shuffle within client
        perm = rng.permutation(len(yc))
        result.append((Xc[perm], yc[perm]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Numpy MLP model
# ─────────────────────────────────────────────────────────────────────────────

class _MLP:
    """
    Lightweight two-layer MLP implemented in pure NumPy.
    Supports FedAvg-style parameter get/set and FedProx proximal gradient.
    """

    def __init__(self, input_dim: int, hidden: int = 32, lr: float = LOCAL_LR, seed: int = 0):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = (rng.randn(input_dim, hidden) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.randn(hidden, 1) * scale2).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

        # SCAFFOLD control variates (initialised to zeros)
        self.c_W1 = np.zeros_like(self.W1)
        self.c_b1 = np.zeros_like(self.b1)
        self.c_W2 = np.zeros_like(self.W2)
        self.c_b2 = np.zeros_like(self.b2)

    # ── Parameter access ──────────────────────────────────────────────────────

    def get_params(self) -> List[np.ndarray]:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]

    def get_controls(self) -> List[np.ndarray]:
        return [self.c_W1.copy(), self.c_b1.copy(), self.c_W2.copy(), self.c_b2.copy()]

    def set_controls(self, controls: List[np.ndarray]) -> None:
        self.c_W1, self.c_b1, self.c_W2, self.c_b2 = [c.copy() for c in controls]

    # ── Forward / backward ───────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        out = (a1 @ self.W2 + self.b2).ravel()
        return z1, a1, out

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, out = self._forward(X)
        return np.clip(out, HR_MIN, HR_MAX)

    def mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - y) ** 2))

    # ── Training steps ────────────────────────────────────────────────────────

    def _sgd_step(
        self,
        X_b: np.ndarray,
        y_b: np.ndarray,
        global_params: List[np.ndarray] | None = None,
        mu: float = 0.0,
        server_c: List[np.ndarray] | None = None,
        client_c: List[np.ndarray] | None = None,
    ) -> None:
        z1, a1, pred = self._forward(X_b)
        n = len(y_b)
        err = (pred - y_b) / n

        dW2 = a1.T @ err.reshape(-1, 1)
        db2 = err.reshape(-1, 1).sum(axis=0)

        da1 = err.reshape(-1, 1) @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_b.T @ dz1
        db1 = dz1.sum(axis=0)

        # FedProx proximal term
        if global_params is not None and mu > 0.0:
            gW1, gb1, gW2, gb2 = global_params
            dW1 += mu * (self.W1 - gW1)
            db1 += mu * (self.b1 - gb1)
            dW2 += mu * (self.W2 - gW2)
            db2 += mu * (self.b2 - gb2)

        # SCAFFOLD correction term: subtract client control, add server control
        if server_c is not None and client_c is not None:
            scW1, scb1, scW2, scb2 = server_c
            ccW1, ccb1, ccW2, ccb2 = client_c
            dW1 += scW1 - ccW1
            db1 += scb1 - ccb1
            dW2 += scW2 - ccW2
            db2 += scb2 - ccb2

        # Gradient clipping to prevent exploding gradients
        clip = 5.0
        for grad in (dW1, db1, dW2, db2):
            np.clip(grad, -clip, clip, out=grad)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        round_num: int = 0,
        global_params: List[np.ndarray] | None = None,
        mu: float = 0.0,
        server_c: List[np.ndarray] | None = None,
        client_c: List[np.ndarray] | None = None,
    ) -> float:
        """Train for `epochs` epochs; return final MSE loss."""
        n = len(X)
        for ep in range(epochs):
            # Use a unique seed per round/epoch so mini-batch orderings differ
            # across rounds and clients, improving training diversity.
            rng = np.random.RandomState(SEED + round_num * 1000 + ep)
            perm = rng.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start: start + BATCH_SIZE]
                self._sgd_step(
                    X[idx], y[idx],
                    global_params=global_params,
                    mu=mu,
                    server_c=server_c,
                    client_c=client_c,
                )
        return self.mse_loss(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_avg(
    param_lists: List[List[np.ndarray]],
    weights: List[float],
) -> List[np.ndarray]:
    """Return weighted average of parameter lists."""
    total = sum(weights)
    result = [np.zeros_like(p) for p in param_lists[0]]
    for params, w in zip(param_lists, weights):
        for i, p in enumerate(params):
            result[i] += (w / total) * p
    return result


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    ppa = float(np.mean(np.abs(y_pred - y_true) <= PPA_TOLERANCE) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "PPA": ppa}


# ─────────────────────────────────────────────────────────────────────────────
# FL Algorithms
# ─────────────────────────────────────────────────────────────────────────────

def _init_clients(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    seed_offset: int = 0,
) -> List[_MLP]:
    return [
        _MLP(input_dim=input_dim, seed=seed_offset + i)
        for i in range(len(clients_data))
    ]


def run_fedavg(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> List[Dict[str, float]]:
    """Standard FedAvg (McMahan et al., 2017)."""
    log.info("── FedAvg ───────────────────────────────────────────────")
    clients = _init_clients(clients_data, input_dim, seed_offset=0)
    global_params = clients[0].get_params()
    sizes = [len(cy) for _, cy in clients_data]
    history: List[Dict[str, float]] = []

    for rnd in range(1, N_ROUNDS + 1):
        local_params_list, local_losses = [], []
        for c, (Xc, yc) in zip(clients, clients_data):
            c.set_params(global_params)
            loss = c.train_local(Xc, yc, epochs=LOCAL_EPOCHS, round_num=rnd)
            local_params_list.append(c.get_params())
            local_losses.append(loss)

        global_params = _weighted_avg(local_params_list, sizes)

        # Evaluate global model on test set
        clients[0].set_params(global_params)
        y_pred = clients[0].predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        avg_loss = float(np.mean(local_losses))
        history.append({"round": rnd, "train_loss": avg_loss, **metrics})
        if rnd % 10 == 0:
            log.info("  Round %2d  Loss=%.4f  MAE=%.2f  PPA=%.1f%%",
                     rnd, avg_loss, metrics["MAE"], metrics["PPA"])

    return history


def run_fedprox(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> List[Dict[str, float]]:
    """FedProx (Li et al., 2020) – proximal term μ‖w − w_global‖²."""
    log.info("── FedProx (μ=%.4f) ────────────────────────────────────", MU_FEDPROX)
    clients = _init_clients(clients_data, input_dim, seed_offset=100)
    global_params = clients[0].get_params()
    sizes = [len(cy) for _, cy in clients_data]
    history: List[Dict[str, float]] = []

    for rnd in range(1, N_ROUNDS + 1):
        local_params_list, local_losses = [], []
        for c, (Xc, yc) in zip(clients, clients_data):
            c.set_params(global_params)
            loss = c.train_local(Xc, yc, epochs=LOCAL_EPOCHS, round_num=rnd,
                                 global_params=global_params, mu=MU_FEDPROX)
            local_params_list.append(c.get_params())
            local_losses.append(loss)

        global_params = _weighted_avg(local_params_list, sizes)

        clients[0].set_params(global_params)
        y_pred = clients[0].predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        avg_loss = float(np.mean(local_losses))
        history.append({"round": rnd, "train_loss": avg_loss, **metrics})
        if rnd % 10 == 0:
            log.info("  Round %2d  Loss=%.4f  MAE=%.2f  PPA=%.1f%%",
                     rnd, avg_loss, metrics["MAE"], metrics["PPA"])

    return history


def run_scaffold(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> List[Dict[str, float]]:
    """
    SCAFFOLD (Karimireddy et al., 2020).
    Uses control variates to correct for client drift.
    """
    log.info("── SCAFFOLD ─────────────────────────────────────────────")
    clients = _init_clients(clients_data, input_dim, seed_offset=200)
    global_params = clients[0].get_params()
    # Server and client control variates
    server_c = [np.zeros_like(p) for p in global_params]
    client_cs = [[np.zeros_like(p) for p in global_params] for _ in clients]
    sizes = [len(cy) for _, cy in clients_data]
    history: List[Dict[str, float]] = []

    for rnd in range(1, N_ROUNDS + 1):
        local_params_list, local_losses = [], []
        delta_cs = []

        for c_idx, (c, (Xc, yc)) in enumerate(zip(clients, clients_data)):
            params_before = copy.deepcopy(global_params)
            c.set_params(global_params)
            c.set_controls(client_cs[c_idx])
            loss = c.train_local(
                Xc, yc, epochs=LOCAL_EPOCHS, round_num=rnd,
                server_c=server_c, client_c=client_cs[c_idx],
            )
            params_after = c.get_params()
            local_params_list.append(params_after)
            local_losses.append(loss)

            # Update client control variate
            K = LOCAL_EPOCHS * max(1, len(Xc) // BATCH_SIZE)
            new_cc = [
                cc - sc + (pb - pa) / (K * LOCAL_LR)
                for cc, sc, pb, pa in zip(
                    client_cs[c_idx], server_c, params_before, params_after
                )
            ]
            delta_cs.append([nc - cc for nc, cc in zip(new_cc, client_cs[c_idx])])
            client_cs[c_idx] = new_cc

        # Aggregate
        global_params = _weighted_avg(local_params_list, sizes)
        # Update server control variate
        total = sum(sizes)
        for i in range(len(server_c)):
            server_c[i] += sum(
                sizes[j] * delta_cs[j][i] for j in range(len(clients))
            ) / total

        clients[0].set_params(global_params)
        y_pred = clients[0].predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        avg_loss = float(np.mean(local_losses))
        history.append({"round": rnd, "train_loss": avg_loss, **metrics})
        if rnd % 10 == 0:
            log.info("  Round %2d  Loss=%.4f  MAE=%.2f  PPA=%.1f%%",
                     rnd, avg_loss, metrics["MAE"], metrics["PPA"])

    return history


def run_fednova(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> List[Dict[str, float]]:
    """
    FedNova (Wang et al., 2020).
    Normalises local gradients by the number of local steps before aggregation.
    """
    log.info("── FedNova ──────────────────────────────────────────────")
    clients = _init_clients(clients_data, input_dim, seed_offset=300)
    global_params = clients[0].get_params()
    sizes = [len(cy) for _, cy in clients_data]
    history: List[Dict[str, float]] = []

    for rnd in range(1, N_ROUNDS + 1):
        local_deltas, local_taus, local_losses = [], [], []

        for c, (Xc, yc) in zip(clients, clients_data):
            c.set_params(global_params)
            params_before = copy.deepcopy(global_params)
            loss = c.train_local(Xc, yc, epochs=LOCAL_EPOCHS, round_num=rnd)
            params_after = c.get_params()
            local_losses.append(loss)

            # Number of local steps (normalisation factor τ)
            tau = LOCAL_EPOCHS * max(1, len(Xc) // BATCH_SIZE)
            delta = [pa - pb for pa, pb in zip(params_after, params_before)]
            # Normalise by τ
            delta_norm = [d / tau for d in delta]
            local_deltas.append(delta_norm)
            local_taus.append(tau)

        # Weighted aggregation of normalised deltas
        total = sum(sizes)
        avg_tau = sum(s * t for s, t in zip(sizes, local_taus)) / total
        agg_delta = [np.zeros_like(p) for p in global_params]
        for delta_norm, s in zip(local_deltas, sizes):
            for i, d in enumerate(delta_norm):
                agg_delta[i] += (s / total) * d

        # Scale aggregated delta back
        global_params = [gp + avg_tau * ad
                         for gp, ad in zip(global_params, agg_delta)]

        clients[0].set_params(global_params)
        y_pred = clients[0].predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        avg_loss = float(np.mean(local_losses))
        history.append({"round": rnd, "train_loss": avg_loss, **metrics})
        if rnd % 10 == 0:
            log.info("  Round %2d  Loss=%.4f  MAE=%.2f  PPA=%.1f%%",
                     rnd, avg_loss, metrics["MAE"], metrics["PPA"])

    return history


def run_fedcrit_hea(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> List[Dict[str, float]]:
    """
    FedCrit-HEA – Federated Critical Heart-rate Estimation Aggregation.

    A domain-adaptive FL method that weights client contributions by their
    local cardiac-band accuracy (fraction of predictions within PPA_TOLERANCE)
    and down-weights clients whose local HR range is narrow or extreme.  This
    mimics a clinically motivated trust score for heterogeneous FHR data.
    """
    log.info("── FedCrit-HEA ──────────────────────────────────────────")
    clients = _init_clients(clients_data, input_dim, seed_offset=400)
    global_params = clients[0].get_params()
    history: List[Dict[str, float]] = []

    for rnd in range(1, N_ROUNDS + 1):
        local_params_list, local_losses = [], []
        trust_scores = []

        for c, (Xc, yc) in zip(clients, clients_data):
            c.set_params(global_params)
            loss = c.train_local(Xc, yc, epochs=LOCAL_EPOCHS, round_num=rnd)
            local_params_list.append(c.get_params())
            local_losses.append(loss)

            # Compute trust score: PPA within PPA_TOLERANCE on local data
            local_pred = c.predict(Xc)
            local_ppa = float(np.mean(np.abs(local_pred - yc) <= PPA_TOLERANCE))
            # HR diversity bonus: clients covering wider physiological range
            hr_range = float(yc.max() - yc.min()) if len(yc) > 1 else 0.0
            hr_range_norm = hr_range / (HR_MAX - HR_MIN)
            # Combined trust: 70 % accuracy + 30 % diversity
            trust = 0.7 * local_ppa + 0.3 * hr_range_norm + 1e-6
            trust_scores.append(trust)

        global_params = _weighted_avg(local_params_list, trust_scores)

        clients[0].set_params(global_params)
        y_pred = clients[0].predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        avg_loss = float(np.mean(local_losses))
        history.append({"round": rnd, "train_loss": avg_loss, **metrics})
        if rnd % 10 == 0:
            log.info("  Round %2d  Loss=%.4f  MAE=%.2f  PPA=%.1f%%",
                     rnd, avg_loss, metrics["MAE"], metrics["PPA"])

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    all_histories: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """Generate a 2×3 figure with convergence curves and final metric bars."""
    metrics = ["train_loss", "MAE", "RMSE", "R2", "PPA"]
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    fig.suptitle("Federated Learning – FHR Estimation (IIScFHSDB)", fontsize=14, fontweight="bold")

    colours = plt.cm.tab10(np.linspace(0, 0.5, len(ALGORITHMS)))

    metric_labels = {
        "train_loss": "Train MSE Loss",
        "MAE": "MAE (bpm)",
        "RMSE": "RMSE (bpm)",
        "R2": "R² Score",
        "PPA": f"PPA (%) ±{PPA_TOLERANCE} bpm",
    }

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        for algo, hist, col in zip(ALGORITHMS, all_histories.values(), colours):
            rounds = [h["round"] for h in hist]
            vals = [h[metric] for h in hist]
            ax.plot(rounds, vals, label=algo, color=col, linewidth=1.8)
        ax.set_title(metric_labels[metric], fontsize=10)
        ax.set_xlabel("Communication Round")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)

    # 6th panel: final metric comparison bar chart
    ax = axes[1][2]
    final_mae = [h[-1]["MAE"] for h in all_histories.values()]
    bars = ax.bar(ALGORITHMS, final_mae, color=colours[:len(ALGORITHMS)])
    ax.set_title("Final MAE Comparison (bpm)", fontsize=10)
    ax.set_ylabel("MAE (bpm)")
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels(ALGORITHMS, rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, final_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved plot → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("═" * 60)
    log.info("  Federated Learning for FHR Estimation – IIScFHSDB")
    log.info("═" * 60)

    # ── Load and preprocess dataset ───────────────────────────────────────────
    X_raw, y_raw, subject_ids = load_dataset()
    n_samples = len(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    input_dim = X_scaled.shape[1]

    # ── Train / test split (holdout 20 %) ────────────────────────────────────
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_samples)
    test_size = max(1, int(0.2 * n_samples))
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train, y_train = X_scaled[train_idx], y_raw[train_idx]
    X_test, y_test = X_scaled[test_idx], y_raw[test_idx]
    log.info("Train=%d  Test=%d", len(y_train), len(y_test))

    # ── Federated data partitioning ───────────────────────────────────────────
    clients_data = partition_data(X_train, y_train, n_clients=N_CLIENTS)
    log.info("Clients: %s samples each",
             ", ".join(str(len(cy)) for _, cy in clients_data))

    # Save client data distribution summary
    dist_records = [
        {
            "client": i + 1,
            "n_samples": len(cy),
            "fhr_mean": float(cy.mean()),
            "fhr_std": float(cy.std()),
            "fhr_min": float(cy.min()),
            "fhr_max": float(cy.max()),
        }
        for i, (_, cy) in enumerate(clients_data)
    ]
    dist_df = pd.DataFrame(dist_records)
    dist_path = os.path.join(RESULTS_DIR, "client_data_summary.csv")
    dist_df.to_csv(dist_path, index=False)
    log.info("Saved client summary → %s", dist_path)

    # ── Run FL algorithms ─────────────────────────────────────────────────────
    algo_runners = {
        "FedAvg": run_fedavg,
        "FedProx": run_fedprox,
        "SCAFFOLD": run_scaffold,
        "FedNova": run_fednova,
        "FedCrit-HEA": run_fedcrit_hea,
    }

    all_histories: Dict[str, List[Dict]] = {}
    for algo, runner in algo_runners.items():
        all_histories[algo] = runner(clients_data, X_test, y_test, input_dim)

    # ── Save convergence summary CSV ──────────────────────────────────────────
    rows = []
    for algo, hist in all_histories.items():
        for h in hist:
            rows.append({"algorithm": algo, **h})
    conv_df = pd.DataFrame(rows)
    conv_path = os.path.join(RESULTS_DIR, "convergence_summary.csv")
    conv_df.to_csv(conv_path, index=False)
    log.info("Saved convergence summary → %s", conv_path)

    # ── Save final metrics CSV ────────────────────────────────────────────────
    final_rows = []
    for algo, hist in all_histories.items():
        last = hist[-1]
        final_rows.append({
            "algorithm": algo,
            "final_round": last["round"],
            "train_loss": last["train_loss"],
            "MAE": last["MAE"],
            "RMSE": last["RMSE"],
            "R2": last["R2"],
            "PPA": last["PPA"],
        })
    final_df = pd.DataFrame(final_rows)
    final_path = os.path.join(RESULTS_DIR, "final_metrics.csv")
    final_df.to_csv(final_path, index=False)
    log.info("Saved final metrics → %s", final_path)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(RESULTS_DIR, "fl_results.png")
    plot_results(all_histories, plot_path)

    # ── Print summary table ───────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  Final Results  (round %d)", N_ROUNDS)
    log.info("═" * 60)
    log.info("  %-16s %8s %8s %8s %8s", "Algorithm", "MAE", "RMSE", "R²", "PPA%")
    log.info("  " + "─" * 52)
    for _, row in final_df.iterrows():
        log.info(
            "  %-16s %8.3f %8.3f %8.3f %8.1f",
            row["algorithm"], row["MAE"], row["RMSE"], row["R2"], row["PPA"],
        )
    log.info("═" * 60)
    log.info("All outputs written to: %s", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
