"""
Deep Learning Model for Fetal Heart Rate Prediction
Dataset: IISc_features_complete_subset.csv
Architectures: Dense NN with Dropout, Deep NN with Batch Normalization, Ensemble
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "IISc_features_complete_subset.csv"
MODELS_DIR = "saved_models"
PLOTS_DIR = "plots"
REPORT_PATH = "model_report.json"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str):
    """Load CSV, drop Subject column, impute NaNs, create stratified splits."""
    df = pd.read_csv(path)

    # Drop identifier column
    df = df.drop(columns=["Subject"], errors="ignore")

    # Separate features and target
    target_col = "Heart_Rate"
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Impute any remaining NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # 70 / 15 / 15 split (stratified by binned target)
    bins = np.percentile(y, [33, 66])
    y_bin = np.digitize(y, bins)

    X_train, X_temp, y_train, y_temp, yb_train, yb_temp = train_test_split(
        X, y, y_bin, test_size=0.30, random_state=SEED, stratify=y_bin
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED,
        stratify=yb_temp
    )

    # Normalise features using training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Data split — train: {X_train.shape[0]}, "
          f"val: {X_val.shape[0]}, test: {X_test.shape[0]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df.drop(columns=[target_col]).columns.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute R², MAE, RMSE, MAPE, and PPA."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE — guard against zero targets
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    # PPA (Peak-to-Peak Amplitude of residuals)
    residuals = y_true - y_pred
    ppa = float(np.max(residuals) - np.min(residuals))

    metrics = {"R2": round(float(r2), 4), "MAE": round(float(mae), 4),
               "RMSE": round(float(rmse), 4), "MAPE": round(float(mape), 4), "PPA": round(float(ppa), 4)}
    if label:
        print(f"  [{label}] R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"MAPE={mape:.4f}%  PPA={ppa:.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODEL ARCHITECTURES
# ──────────────────────────────────────────────────────────────────────────────

def build_dense_dropout(input_dim: int, units: int = 128, dropout: float = 0.3,
                        lr: float = 1e-3, l2: float = 1e-4) -> keras.Model:
    """Dense NN with dropout regularisation."""
    inp = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 2, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 4, activation="relu")(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inp, out, name="dense_dropout")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="mse", metrics=["mae"])
    return model


def build_batchnorm_deep(input_dim: int, units: int = 256, dropout: float = 0.2,
                         lr: float = 5e-4, l2: float = 1e-4) -> keras.Model:
    """Deep NN with batch normalisation."""
    inp = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(units // 2, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(units // 4, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(units // 8, activation="relu")(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inp, out, name="batchnorm_deep")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="mse", metrics=["mae"])
    return model


def build_residual(input_dim: int, units: int = 128, dropout: float = 0.25,
                   lr: float = 1e-3) -> keras.Model:
    """Residual / skip-connection network."""
    inp = keras.Input(shape=(input_dim,), name="input")

    # Projection to common dimension
    proj = layers.Dense(units, activation="relu", name="projection")(inp)

    # Block 1
    x = layers.Dense(units, activation="relu")(proj)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Add()([x, proj])

    # Block 2
    x2 = layers.Dense(units // 2, activation="relu")(x)
    x2 = layers.Dropout(dropout)(x2)

    out = layers.Dense(1, name="output")(x2)

    model = keras.Model(inp, out, name="residual_net")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="mse", metrics=["mae"])
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(name: str, patience: int = 30) -> list:
    ckpt_path = os.path.join(MODELS_DIR, f"{name}_best.keras")
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15,
                          min_lr=1e-6, verbose=0),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]


def train_model(model: keras.Model, name: str,
                train_data, val_data,
                batch_size: int = 16, epochs: int = 500) -> keras.callbacks.History:
    X_train, y_train = train_data
    X_val, y_val = val_data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=get_callbacks(name),
        verbose=0,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER OPTIMISATION  (lightweight grid search)
# ──────────────────────────────────────────────────────────────────────────────

def hyperparameter_search(input_dim: int, train_data, val_data,
                          n_top: int = 1) -> dict:
    """Grid search over a small parameter grid; returns best config."""
    param_grid = {
        "units":    [64, 128, 256],
        "dropout":  [0.2, 0.3],
        "lr":       [1e-3, 5e-4],
        "batch_size": [8, 16],
    }

    X_train, y_train = train_data
    X_val, y_val = val_data

    best_val_loss = np.inf
    best_cfg = {}
    results = []

    combos = list(product(param_grid["units"], param_grid["dropout"],
                          param_grid["lr"], param_grid["batch_size"]))

    print(f"\nGrid search over {len(combos)} combinations …")
    for units, dropout, lr, bs in combos:
        cfg_name = f"gs_u{units}_d{int(dropout*10)}_lr{int(lr*1e4)}_bs{bs}"
        model = build_batchnorm_deep(input_dim, units=units,
                                     dropout=dropout, lr=lr)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=bs, epochs=300,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=25,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=12, min_lr=1e-6, verbose=0),
            ],
            verbose=0,
        )
        val_loss = float(min(history.history["val_loss"]))
        results.append({"units": units, "dropout": dropout,
                        "lr": float(lr), "batch_size": bs, "val_mse": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cfg = {"units": units, "dropout": dropout,
                        "lr": lr, "batch_size": bs}
        print(f"  {cfg_name}: val_mse={val_loss:.4f}")

    print(f"\nBest config: {best_cfg}  (val_mse={best_val_loss:.4f})")
    return best_cfg, results


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ENSEMBLE PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """Simple average ensemble of multiple Keras models."""

    def __init__(self, models: list, weights: list = None):
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = np.array(weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X, verbose=0).ravel()
                          for m in self.models], axis=1)
        return (preds * self.weights).sum(axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, (name, hist) in zip(axes[0], histories.items()):
        ax.plot(hist.history["loss"], label="train loss")
        ax.plot(hist.history["val_loss"], label="val loss")
        ax.set_title(f"{name} – Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_predictions(y_true: np.ndarray, predictions: dict, split_name: str):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
        lo = min(y_true.min(), y_pred.min()) - 2
        hi = max(y_true.max(), y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{name} [{split_name}]  R²={r2:.3f}")
        ax.set_xlabel("Actual Heart Rate (bpm)")
        ax.set_ylabel("Predicted Heart Rate (bpm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"predictions_{split_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_residuals(y_true: np.ndarray, predictions: dict, split_name: str):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, (name, y_pred) in zip(axes[0], predictions.items()):
        residuals = y_true - y_pred
        ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
        ax.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", linewidths=0.4)
        ax.set_title(f"{name} [{split_name}] Residuals")
        ax.set_xlabel("Predicted Heart Rate (bpm)")
        ax.set_ylabel("Residual (bpm)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"residuals_{split_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_metrics_bar(all_metrics: dict):
    """Bar chart comparing all models across all metrics."""
    metric_names = ["R2", "MAE", "RMSE", "MAPE", "PPA"]
    splits = ["train", "val", "test"]
    models = list(all_metrics.keys())

    fig, axes = plt.subplots(1, len(metric_names),
                             figsize=(4 * len(metric_names), 5))
    x = np.arange(len(models))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric in zip(axes, metric_names):
        for i, split in enumerate(splits):
            vals = [all_metrics[m][split][metric] for m in models]
            ax.bar(x + i * width, vals, width, label=split, color=colors[i])
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison — All Metrics", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Fetal Heart Rate Deep Learning Pipeline")
    print("=" * 65)

    # ── 8a. Load data ─────────────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = \
        load_and_preprocess(DATA_PATH)
    input_dim = X_train.shape[1]

    # ── 8b. Hyperparameter optimisation ───────────────────────────
    best_cfg, gs_results = hyperparameter_search(
        input_dim, (X_train, y_train), (X_val, y_val)
    )

    # ── 8c. Train individual models with best / default configs ───
    print("\n" + "─" * 65)
    print("Training Model 1: Dense + Dropout (optimised config)")
    model_dd = build_dense_dropout(
        input_dim,
        units=best_cfg["units"],
        dropout=best_cfg["dropout"],
        lr=best_cfg["lr"],
    )
    hist_dd = train_model(model_dd, "dense_dropout",
                          (X_train, y_train), (X_val, y_val),
                          batch_size=best_cfg["batch_size"])

    print("Training Model 2: Deep + BatchNorm (optimised config)")
    model_bn = build_batchnorm_deep(
        input_dim,
        units=best_cfg["units"],
        dropout=best_cfg["dropout"],
        lr=best_cfg["lr"],
    )
    hist_bn = train_model(model_bn, "batchnorm_deep",
                          (X_train, y_train), (X_val, y_val),
                          batch_size=best_cfg["batch_size"])

    print("Training Model 3: Residual Network")
    model_res = build_residual(input_dim, units=best_cfg["units"],
                               dropout=best_cfg["dropout"], lr=best_cfg["lr"])
    hist_res = train_model(model_res, "residual_net",
                           (X_train, y_train), (X_val, y_val),
                           batch_size=best_cfg["batch_size"])

    # ── 8d. Ensemble ──────────────────────────────────────────────
    # Weight models by inverse val-loss (better model → higher weight)
    val_losses = [
        min(hist_dd.history["val_loss"]),
        min(hist_bn.history["val_loss"]),
        min(hist_res.history["val_loss"]),
    ]
    inv_losses = [1.0 / v for v in val_losses]
    total = sum(inv_losses)
    ensemble_weights = [w / total for w in inv_losses]
    ensemble = EnsembleModel(
        [model_dd, model_bn, model_res], weights=ensemble_weights
    )
    print(f"\nEnsemble weights: dd={ensemble_weights[0]:.3f}  "
          f"bn={ensemble_weights[1]:.3f}  res={ensemble_weights[2]:.3f}")

    # ── 8e. Evaluate ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("Evaluation Metrics")
    print("─" * 65)

    model_names = ["dense_dropout", "batchnorm_deep", "residual_net", "ensemble"]
    predictors = {
        "dense_dropout": lambda X: model_dd.predict(X, verbose=0).ravel(),
        "batchnorm_deep": lambda X: model_bn.predict(X, verbose=0).ravel(),
        "residual_net":   lambda X: model_res.predict(X, verbose=0).ravel(),
        "ensemble":       lambda X: ensemble.predict(X),
    }

    all_metrics = {}
    for mname, pred_fn in predictors.items():
        print(f"\n  {mname}")
        all_metrics[mname] = {
            "train": compute_metrics(y_train, pred_fn(X_train), "train"),
            "val":   compute_metrics(y_val,   pred_fn(X_val),   "val"),
            "test":  compute_metrics(y_test,  pred_fn(X_test),  "test"),
        }

    # ── 8f. Visualisations ────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_curves({"dense_dropout": hist_dd,
                          "batchnorm_deep": hist_bn,
                          "residual_net": hist_res})

    for split_name, X_s, y_s in [("train", X_train, y_train),
                                   ("val",   X_val,   y_val),
                                   ("test",  X_test,  y_test)]:
        preds = {n: pred_fn(X_s) for n, pred_fn in predictors.items()}
        plot_predictions(y_s, preds, split_name)
        plot_residuals(y_s, preds, split_name)

    plot_metrics_bar(all_metrics)

    # ── 8g. Save models ───────────────────────────────────────────
    print("\nSaving models …")
    model_dd.save(os.path.join(MODELS_DIR, "dense_dropout.keras"))
    model_bn.save(os.path.join(MODELS_DIR, "batchnorm_deep.keras"))
    model_res.save(os.path.join(MODELS_DIR, "residual_net.keras"))

    # SavedModel format
    model_dd.export(os.path.join(MODELS_DIR, "dense_dropout_savedmodel"))
    model_bn.export(os.path.join(MODELS_DIR, "batchnorm_deep_savedmodel"))
    model_res.export(os.path.join(MODELS_DIR, "residual_net_savedmodel"))

    # Scaler params (for inference)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
    }
    with open(os.path.join(MODELS_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)

    # ── 8h. Summary report ────────────────────────────────────────
    report = {
        "best_hyperparameters": best_cfg,
        "grid_search_results": gs_results,
        "model_metrics": all_metrics,
        "ensemble_weights": {
            "dense_dropout":  ensemble_weights[0],
            "batchnorm_deep": ensemble_weights[1],
            "residual_net":   ensemble_weights[2],
        },
    }
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, cls=_NpEncoder)
    print(f"Saved report: {REPORT_PATH}")

    # ── 8i. Pretty summary table ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    header = f"{'Model':<18} {'Split':<6} {'R²':>7} {'MAE':>7} {'RMSE':>7} {'MAPE%':>8} {'PPA':>8}"
    print(header)
    print("─" * len(header))
    for mname in model_names:
        for split in ["train", "val", "test"]:
            m = all_metrics[mname][split]
            print(f"{mname:<18} {split:<6} {m['R2']:>7.4f} {m['MAE']:>7.4f} "
                  f"{m['RMSE']:>7.4f} {m['MAPE']:>8.4f} {m['PPA']:>8.4f}")
        print()

    print("Pipeline complete. Artefacts saved in:")
    print(f"  Models : {MODELS_DIR}/")
    print(f"  Plots  : {PLOTS_DIR}/")
    print(f"  Report : {REPORT_PATH}")


if __name__ == "__main__":
    main()
