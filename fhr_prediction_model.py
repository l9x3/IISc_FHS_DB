"""
Fetal Heart Rate (FHR) Prediction Deep Learning Model
======================================================
Dataset: Records.csv - 60 pregnant women records (44 with FHR measurements)
Reference: https://www.physionet.org/content/fetalheartsounddata/1.0/

Architecture:
  Input (8 features) → Dense(64, ReLU) + BatchNorm + Dropout
                     → Dense(32, ReLU) + BatchNorm + Dropout
                     → Dense(16, ReLU) + BatchNorm + Dropout
                     → Dense(1, Sigmoid) scaled to [110, 160] bpm
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
FHR_MIN, FHR_MAX = 110.0, 160.0
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 20
BATCH_SIZE = 16
MAX_EPOCHS = 300

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# Clinical risk mapping
# ─────────────────────────────────────────────────────────────
CLINICAL_RISK_MAP = {
    'normal':               0.0,
    'anaemia':              0.6,
    'anemia':               0.6,
    'iugr':                 0.8,
    'intrauterine growth':  0.8,
    'preeclampsia':         0.9,
    'pre-eclampsia':        0.9,
    'gestational diabetes': 0.7,
    'hypertension':         0.85,
    'placenta previa':      0.95,
    'oligohydramnios':      0.75,
    'polyhydramnios':       0.65,
    'hypothyroidism':       0.55,
}


def parse_clinical_risk(condition_str: str) -> float:
    """Return max risk score for the given clinical condition string."""
    if pd.isna(condition_str) or str(condition_str).strip() == '':
        return 0.0
    condition_lower = str(condition_str).lower()
    max_risk = 0.0
    for keyword, risk in CLINICAL_RISK_MAP.items():
        if keyword in condition_lower:
            max_risk = max(max_risk, risk)
    return max_risk


# ─────────────────────────────────────────────────────────────
# Data loading & feature engineering
# ─────────────────────────────────────────────────────────────
def load_and_engineer_features(csv_path: str):
    """
    Load Records.csv and engineer 8 clinical features.

    Features:
        1. Gestational age (weeks)
        2. Maternal age (years)
        3. Weight (kg)     – median imputation for missing
        4. Height (cm)     – median imputation for missing
        5. Gravida         – median imputation for missing
        6. Para            – median imputation for missing
        7. BMI             – weight / (height_m)²
        8. Clinical Risk Score

    Returns
    -------
    X_all : np.ndarray  (N_all, 8)   – all 60 rows
    y_all : np.ndarray  (N_fhr,)     – FHR values for rows that have them
    df    : pd.DataFrame             – engineered dataframe
    fhr_mask : boolean series        – which rows have FHR
    feature_names : list[str]
    """
    df = pd.read_csv(csv_path)

    # ── median imputation for numeric columns ──────────────────
    for col in ['Weight_kg', 'Height_cm', 'Gravida', 'Para']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    for col in ['Gestational_Age_Weeks', 'Maternal_Age_Years']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── BMI ────────────────────────────────────────────────────
    df['BMI'] = df['Weight_kg'] / (df['Height_cm'] / 100.0) ** 2

    # ── Clinical risk score ────────────────────────────────────
    cond_col = next(
        (c for c in df.columns if 'condition' in c.lower() or 'clinical' in c.lower()),
        None
    )
    if cond_col:
        df['Clinical_Risk_Score'] = df[cond_col].apply(parse_clinical_risk)
    else:
        df['Clinical_Risk_Score'] = 0.0

    feature_names = [
        'Gestational_Age_Weeks',
        'Maternal_Age_Years',
        'Weight_kg',
        'Height_cm',
        'Gravida',
        'Para',
        'BMI',
        'Clinical_Risk_Score',
    ]

    # ── FHR target ─────────────────────────────────────────────
    df['FHR_bpm'] = pd.to_numeric(df.get('FHR_bpm', np.nan), errors='coerce')
    fhr_mask = df['FHR_bpm'].notna()

    X_all = df[feature_names].values.astype(np.float32)
    y_fhr = df.loc[fhr_mask, 'FHR_bpm'].values.astype(np.float32)
    X_fhr = df.loc[fhr_mask, feature_names].values.astype(np.float32)

    print(f"\n{'='*60}")
    print("  Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total records        : {len(df)}")
    print(f"  Records with FHR     : {fhr_mask.sum()}")
    print(f"  FHR range            : {y_fhr.min():.1f} – {y_fhr.max():.1f} bpm")
    print(f"  Features engineered  : {len(feature_names)}")
    print(f"{'='*60}\n")

    return X_fhr, y_fhr, df, fhr_mask, feature_names


# ─────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────
def build_model(n_features: int, dropout_rate: float = 0.3) -> keras.Model:
    """
    Neural network regression model:
        Input → Dense(64) → BN → Dropout
              → Dense(32) → BN → Dropout
              → Dense(16) → BN → Dropout
              → Dense(1, sigmoid) scaled to [FHR_MIN, FHR_MAX]
    """
    inp = keras.Input(shape=(n_features,), name='clinical_features')

    x = layers.Dense(64, activation='relu', name='dense_64')(inp)
    x = layers.BatchNormalization(name='bn_64')(x)
    x = layers.Dropout(dropout_rate, name='drop_64')(x)

    x = layers.Dense(32, activation='relu', name='dense_32')(x)
    x = layers.BatchNormalization(name='bn_32')(x)
    x = layers.Dropout(dropout_rate, name='drop_32')(x)

    x = layers.Dense(16, activation='relu', name='dense_16')(x)
    x = layers.BatchNormalization(name='bn_16')(x)
    x = layers.Dropout(dropout_rate / 2, name='drop_16')(x)

    # Sigmoid output → scale to [FHR_MIN, FHR_MAX]
    sig_out = layers.Dense(1, activation='sigmoid', name='sigmoid_out')(x)
    out = layers.Lambda(
        lambda t: t * (FHR_MAX - FHR_MIN) + FHR_MIN,
        name='fhr_bpm'
    )(sig_out)

    model = keras.Model(inputs=inp, outputs=out, name='FHR_Predictor')
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, n_features):
    model = build_model(n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )
    return model, history


# ─────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler, feature_names):
    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0
    r2   = r2_score(y_test, y_pred)

    metrics = {
        'MAE (bpm)' : mae,
        'RMSE (bpm)': rmse,
        'MSE'       : mse,
        'MAPE (%)'  : mape,
        'R²'        : r2,
    }

    print(f"\n{'='*60}")
    print("  Test-Set Evaluation Metrics")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Value':>12}")
    print(f"  {'-'*32}")
    for name, val in metrics.items():
        print(f"  {name:<20} {val:>12.4f}")
    print(f"{'='*60}\n")

    return y_pred, metrics


# ─────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────
def plot_training_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('FHR Prediction Model – Training History', fontsize=14, fontweight='bold')

    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history.history['loss'],     label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(epochs, history.history['mae'],     label='Train MAE', linewidth=2, color='orange')
    axes[1].plot(epochs, history.history['val_mae'], label='Val MAE',   linewidth=2, color='red', linestyle='--')
    axes[1].set_title('Mean Absolute Error (bpm)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (bpm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Training history → {save_path}")


def plot_evaluation(y_test, y_pred, metrics, save_path: str):
    residuals = y_test - y_pred
    mape_per  = np.abs((y_test - y_pred) / y_test) * 100.0

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('FHR Prediction – Evaluation Report', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Predictions vs Actual ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, color='steelblue', s=80, edgecolors='navy', alpha=0.8, zorder=3)
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax1.set_xlabel('Actual FHR (bpm)')
    ax1.set_ylabel('Predicted FHR (bpm)')
    ax1.set_title('Predicted vs Actual FHR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Annotate each point
    for i, (a, p) in enumerate(zip(y_test, y_pred)):
        ax1.annotate(f'S{i+1}', (a, p), textcoords='offset points', xytext=(4, 3), fontsize=7)

    # Metrics text box
    metric_txt = '\n'.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    ax1.text(0.03, 0.97, metric_txt, transform=ax1.transAxes,
             fontsize=7, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Panel 2: Residuals ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, len(residuals) + 1), residuals,
            color=['tomato' if r < 0 else 'seagreen' for r in residuals],
            edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Residual (bpm)')
    ax2.set_title('Residuals (Actual − Predicted)')
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Error distribution ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=min(10, len(residuals)), color='steelblue',
             edgecolor='black', alpha=0.75, density=True)
    ax3.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero error')
    ax3.axvline(residuals.mean(), color='orange', linewidth=1.5, linestyle='-.',
                label=f'Mean = {residuals.mean():.2f}')
    ax3.set_xlabel('Residual (bpm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: MAPE per sample ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ['tomato' if m > 5 else 'seagreen' for m in mape_per]
    ax4.bar(range(1, len(mape_per) + 1), mape_per, color=colors, edgecolor='black', linewidth=0.5)
    ax4.axhline(5, color='orange', linewidth=1.5, linestyle='--', label='5% threshold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('MAPE (%)')
    ax4.set_title('MAPE Distribution per Sample')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Evaluation panels → {save_path}")


def plot_feature_importance(model, X_test, y_test, feature_names, save_path: str):
    """Permutation-based feature importance on the test set."""
    baseline_mae = mean_absolute_error(
        y_test, model.predict(X_test, verbose=0).flatten()
    )

    importances = []
    rng = np.random.default_rng(RANDOM_SEED)
    for i, fname in enumerate(feature_names):
        X_permuted = X_test.copy()
        X_permuted[:, i] = rng.permutation(X_permuted[:, i])
        perm_mae = mean_absolute_error(
            y_test, model.predict(X_permuted, verbose=0).flatten()
        )
        importances.append(perm_mae - baseline_mae)

    importances = np.array(importances)
    sorted_idx  = np.argsort(importances)[::-1]
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue' if v >= 0 else 'salmon' for v in sorted_imp]
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1],
                   edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Increase in MAE when feature is permuted (bpm)')
    ax.set_title('Clinical Feature Importance (Permutation-Based)', fontweight='bold')

    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Feature importance → {save_path}")


# ─────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────
def save_predictions_csv(y_test, y_pred, save_path: str):
    df_out = pd.DataFrame({
        'Sample_Index': range(1, len(y_test) + 1),
        'Actual_FHR_bpm': y_test,
        'Predicted_FHR_bpm': np.round(y_pred, 2),
        'Residual_bpm': np.round(y_test - y_pred, 2),
        'MAPE_pct': np.round(np.abs((y_test - y_pred) / y_test) * 100, 2),
    })
    df_out.to_csv(save_path, index=False)
    print(f"  [Saved] Predictions CSV   → {save_path}")


def save_summary_txt(metrics, history, y_test, y_pred, save_path: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_epochs = len(history.history['loss'])

    lines = [
        "=" * 60,
        "  FHR Prediction Model – Evaluation Summary",
        f"  Generated: {now}",
        "=" * 60,
        "",
        "  MODEL ARCHITECTURE",
        "  ------------------",
        "  Input (8 features)",
        "  → Dense(64, ReLU) + BatchNorm + Dropout(0.3)",
        "  → Dense(32, ReLU) + BatchNorm + Dropout(0.3)",
        "  → Dense(16, ReLU) + BatchNorm + Dropout(0.15)",
        "  → Dense(1, Sigmoid) scaled to [110, 160] bpm",
        "",
        "  TRAINING CONFIGURATION",
        "  ----------------------",
        f"  Optimizer       : Adam (lr={LEARNING_RATE})",
        "  Loss function   : MSE",
        f"  Early stopping  : patience={EARLY_STOP_PATIENCE}",
        f"  Epochs trained  : {n_epochs}",
        f"  Batch size      : {BATCH_SIZE}",
        "",
        "  DATA SPLIT",
        "  ----------",
        "  Train / Val / Test : 70% / 15% / 15%",
        f"  Samples (train/val/test): see console output",
        "",
        "  TEST-SET METRICS",
        "  ----------------",
    ]
    for name, val in metrics.items():
        lines.append(f"  {name:<20} {val:.4f}")
    lines += [
        "",
        "  PER-SAMPLE PREDICTIONS",
        "  ----------------------",
        f"  {'Sample':<8} {'Actual':>10} {'Predicted':>12} {'Residual':>10} {'MAPE%':>8}",
        f"  {'-'*52}",
    ]
    for i, (a, p) in enumerate(zip(y_test, y_pred)):
        res  = a - p
        mape = abs(res / a) * 100.0
        lines.append(f"  {i+1:<8} {a:>10.1f} {p:>12.2f} {res:>10.2f} {mape:>8.2f}")
    lines += ["", "=" * 60]

    with open(save_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  [Saved] Summary text      → {save_path}")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def main():
    csv_path = os.path.join(OUTPUT_DIR, 'Records.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Records.csv not found at: {csv_path}")

    # ── 1. Load & engineer features ───────────────────────────
    X, y, df_eng, fhr_mask, feature_names = load_and_engineer_features(csv_path)

    # ── 2. Split: 70 / 15 / 15 ────────────────────────────────
    # With 44 labelled samples: ~30 train / ~7 val / ~7 test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15 / 0.85,      # ≈17.6% of train_val → 15% of total
        random_state=RANDOM_SEED
    )

    print(f"  Split sizes  →  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── 3. Normalise with StandardScaler ──────────────────────
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val   = feature_scaler.transform(X_val)
    X_test  = feature_scaler.transform(X_test)

    # ── 4. Build & train model ────────────────────────────────
    model, history = train_model(X_train, y_train, X_val, y_val, n_features=X_train.shape[1])

    # ── 5. Evaluate on test set ───────────────────────────────
    y_pred, metrics = evaluate_model(model, X_test, y_test, feature_scaler, feature_names)

    # ── 6. Save outputs ───────────────────────────────────────
    print("\n  Saving outputs …")
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, 'fhr_training_history.png'),
    )
    plot_evaluation(
        y_test, y_pred, metrics,
        save_path=os.path.join(OUTPUT_DIR, 'fhr_evaluation_panels.png'),
    )
    plot_feature_importance(
        model, X_test, y_test, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'fhr_feature_importance.png'),
    )
    save_predictions_csv(
        y_test, y_pred,
        save_path=os.path.join(OUTPUT_DIR, 'fhr_predictions.csv'),
    )
    save_summary_txt(
        metrics, history, y_test, y_pred,
        save_path=os.path.join(OUTPUT_DIR, 'fhr_evaluation_summary.txt'),
    )

    print("\n  ✓ All outputs saved successfully.")
    return metrics


if __name__ == '__main__':
    main()
