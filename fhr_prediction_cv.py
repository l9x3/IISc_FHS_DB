"""
Fetal Heart Rate (FHR) Prediction Model with 5-Fold Cross-Validation
Dataset: Records.csv (60 records, 44 with FHR measurements in 130-156 bpm range)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5
EPOCHS = 300
BATCH_SIZE = 8
PATIENCE = 20
FHR_MIN = 110.0
FHR_MAX = 160.0

FEATURE_COLS = [
    'Gestational_Age',
    'Maternal_Age',
    'Weight_kg',
    'Height_cm',
    'Gravida',
    'Para',
    'BMI',
    'Clinical_Risk_Score',
]
TARGET_COL = 'FHR_bpm'

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────

def load_and_preprocess(csv_path: str):
    """Load Records.csv, keep only labelled rows, impute & scale features."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total records.")

    # Keep rows that have a FHR measurement
    df_labelled = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Records with FHR measurements: {len(df_labelled)}")
    print(f"FHR range: {df_labelled[TARGET_COL].min():.1f} – "
          f"{df_labelled[TARGET_COL].max():.1f} bpm\n")

    X_raw = df_labelled[FEATURE_COLS].values.astype(np.float32)
    y = df_labelled[TARGET_COL].values.astype(np.float32)

    # Median imputation for any remaining NaNs in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)

    return X_imputed, y, df_labelled


# ─────────────────────────────────────────────
# 2. Model Builder
# ─────────────────────────────────────────────

def build_model(n_features: int = 8) -> keras.Model:
    """Build the regression model described in the specification."""
    inputs = keras.Input(shape=(n_features,), name='clinical_features')

    # Hidden layer 1: 64 neurons
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Hidden layer 2: 32 neurons
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Hidden layer 3: 16 neurons
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # Output: sigmoid → scale to [FHR_MIN, FHR_MAX]
    sigmoid_out = layers.Dense(1, activation='sigmoid')(x)
    outputs = layers.Lambda(
        lambda t: t * (FHR_MAX - FHR_MIN) + FHR_MIN,
        name='fhr_output'
    )(sigmoid_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae'],
    )
    return model


# ─────────────────────────────────────────────
# 3. Metrics Helper
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape, 'R2': r2}


# ─────────────────────────────────────────────
# 4. 5-Fold Cross-Validation
# ─────────────────────────────────────────────

def run_cross_validation(X: np.ndarray, y: np.ndarray):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []          # one dict per fold
    all_predictions = []       # all (actual, predicted, fold) rows

    print("=" * 65)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 65)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n── Fold {fold_idx} ──────────────────────────────────────────")
        print(f"   Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Per-fold scaling (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build fresh model for each fold
        keras.backend.clear_session()
        model = build_model(n_features=X_train_scaled.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )

        epochs_run = len(history.history['loss'])
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()

        metrics = compute_metrics(y_test, y_pred)
        metrics['Fold'] = fold_idx
        metrics['Epochs'] = epochs_run
        fold_results.append(metrics)

        print(f"   Epochs run: {epochs_run}")
        print(f"   MAE={metrics['MAE']:.3f} bpm | RMSE={metrics['RMSE']:.3f} bpm | "
              f"MSE={metrics['MSE']:.3f} | MAPE={metrics['MAPE']:.2f}% | R²={metrics['R2']:.4f}")

        for act, pred in zip(y_test, y_pred):
            all_predictions.append({'Fold': fold_idx, 'Actual_FHR': act, 'Predicted_FHR': pred})

    return fold_results, all_predictions


# ─────────────────────────────────────────────
# 5. Results Display & Export
# ─────────────────────────────────────────────

def display_results_table(fold_results: list) -> pd.DataFrame:
    metric_keys = ['MAE', 'RMSE', 'MSE', 'MAPE', 'R2']
    df = pd.DataFrame(fold_results)[['Fold', 'Epochs'] + metric_keys]

    # Compute summary rows
    mean_row = {'Fold': 'Mean', 'Epochs': '—'}
    std_row = {'Fold': 'Std', 'Epochs': '—'}
    for k in metric_keys:
        vals = [r[k] for r in fold_results]
        mean_row[k] = np.mean(vals)
        std_row[k] = np.std(vals)

    summary_df = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

    print("\n" + "=" * 65)
    print("RESULTS TABLE")
    print("=" * 65)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    return summary_df


def save_per_fold_csv(fold_results: list, path: str):
    metric_keys = ['Fold', 'Epochs', 'MAE', 'RMSE', 'MSE', 'MAPE', 'R2']
    df = pd.DataFrame(fold_results)[metric_keys]

    mean_row = {'Fold': 'Mean', 'Epochs': '—'}
    std_row = {'Fold': 'Std', 'Epochs': '—'}
    for k in ['MAE', 'RMSE', 'MSE', 'MAPE', 'R2']:
        vals = df[k].values
        mean_row[k] = np.mean(vals)
        std_row[k] = np.std(vals)

    out = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"Per-fold results saved → {path}")


def save_predictions_csv(all_predictions: list, path: str):
    pd.DataFrame(all_predictions).to_csv(path, index=False)
    print(f"All-fold predictions saved → {path}")


def save_summary_report(fold_results: list, path: str):
    metric_keys = ['MAE', 'RMSE', 'MSE', 'MAPE', 'R2']
    lines = []
    lines.append("=" * 65)
    lines.append("FHR PREDICTION MODEL – 5-FOLD CROSS-VALIDATION SUMMARY")
    lines.append("=" * 65)
    lines.append("")
    lines.append("Model Architecture:")
    lines.append("  Input: 8 normalised clinical features")
    lines.append("  Hidden 1: Dense(64, ReLU) → BatchNorm → Dropout(0.20)")
    lines.append("  Hidden 2: Dense(32, ReLU) → BatchNorm → Dropout(0.20)")
    lines.append("  Hidden 3: Dense(16, ReLU) → BatchNorm → Dropout(0.15)")
    lines.append("  Output:   Dense(1, Sigmoid) scaled to [110, 160] bpm")
    lines.append("  Optimizer: Adam (lr=0.001) | Loss: MSE | EarlyStopping(patience=20)")
    lines.append("")
    lines.append("Per-Fold Metrics:")
    lines.append("-" * 65)
    header = f"{'Fold':>6}  {'MAE':>8}  {'RMSE':>8}  {'MSE':>10}  {'MAPE%':>8}  {'R²':>8}  {'Epochs':>7}"
    lines.append(header)
    lines.append("-" * 65)
    for r in fold_results:
        lines.append(
            f"{r['Fold']:>6}  {r['MAE']:>8.3f}  {r['RMSE']:>8.3f}  "
            f"{r['MSE']:>10.3f}  {r['MAPE']:>8.2f}  {r['R2']:>8.4f}  {r['Epochs']:>7}"
        )
    lines.append("-" * 65)
    fmt_map = {'MAE': '8.3f', 'RMSE': '8.3f', 'MSE': '10.3f', 'MAPE': '8.2f', 'R2': '8.4f'}
    for label, func in [('Mean', np.mean), ('Std', np.std)]:
        row = f"{label:>6}"
        for k in metric_keys:
            vals = [r[k] for r in fold_results]
            row += f"  {func(vals):{fmt_map[k]}}"
        lines.append(row)
    lines.append("=" * 65)
    lines.append("")
    lines.append("Dataset Details:")
    lines.append("  Total records: 60")
    lines.append("  Records with FHR: 44")
    lines.append("  FHR range: 130–156 bpm")
    lines.append("  Features: Gestational Age, Maternal Age, Weight, Height,")
    lines.append("            Gravida, Para, BMI, Clinical Risk Score")
    lines.append("  Preprocessing: Median imputation + StandardScaler (per fold)")
    lines.append("")
    lines.append("Cross-Validation Setup:")
    lines.append("  Strategy: KFold (k=5, shuffled, random_state=42)")
    lines.append("  Train/Test split: 80% / 20% per fold")
    lines.append("  Early Stopping: patience=20, monitored on val_loss")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(path, 'w') as f:
        f.write(report_text + "\n")
    print(f"\nSummary report saved → {path}")


# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']


def plot_metrics_bar(fold_results: list, save_dir: str):
    """Bar chart comparing each metric across folds."""
    metric_keys = ['MAE', 'RMSE', 'MSE', 'MAPE', 'R2']
    n_metrics = len(metric_keys)
    folds = [r['Fold'] for r in fold_results]
    x = np.arange(len(folds))

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5))
    fig.suptitle('Metrics Comparison Across Folds', fontsize=14, fontweight='bold')

    for ax, key in zip(axes, metric_keys):
        vals = [r[key] for r in fold_results]
        bars = ax.bar(x, vals, width=0.5, color=COLORS, edgecolor='white', linewidth=0.8)
        ax.set_title(key, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{f}' for f in folds])
        ax.set_xlabel('Fold')
        unit = ' (bpm)' if key in ('MAE', 'RMSE') else (' (bpm²)' if key == 'MSE' else (' (%)' if key == 'MAPE' else ''))
        ax.set_ylabel(key + unit)
        mean_val = np.mean(vals)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f'Mean={mean_val:.3f}')
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fhr_cv_metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved → {path}")


def plot_box_plots(fold_results: list, save_dir: str):
    """Box plots showing metric distributions across folds."""
    metric_keys = ['MAE', 'RMSE', 'MSE', 'MAPE', 'R2']
    data = {k: [r[k] for r in fold_results] for k in metric_keys}
    units = {'MAE': 'bpm', 'RMSE': 'bpm', 'MSE': 'bpm²', 'MAPE': '%', 'R2': ''}

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(18, 5))
    fig.suptitle('Metric Distributions Across Folds (Box Plots)', fontsize=14, fontweight='bold')

    for ax, key in zip(axes, metric_keys):
        bp = ax.boxplot(
            data[key],
            patch_artist=True,
            medianprops={'color': 'red', 'linewidth': 2},
        )
        for patch in bp['boxes']:
            patch.set_facecolor('#AEC6E8')
        ax.set_title(key, fontweight='bold')
        ax.set_xlabel('All Folds')
        unit = units[key]
        ax.set_ylabel(f'{key} ({unit})' if unit else key)
        # Overlay individual fold points
        ax.scatter(
            np.ones(len(data[key])) + np.random.uniform(-0.08, 0.08, len(data[key])),
            data[key],
            color=COLORS[:len(data[key])],
            zorder=5,
            s=50,
            edgecolor='black',
            linewidth=0.5,
        )

    plt.tight_layout()
    path = os.path.join(save_dir, 'fhr_cv_box_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Box plots saved → {path}")


def plot_predictions_vs_actual(all_predictions: list, save_dir: str):
    """Combined scatter plot of predictions vs actuals from all folds."""
    df = pd.DataFrame(all_predictions)

    fig, ax = plt.subplots(figsize=(8, 7))
    for fold_id, color in zip(range(1, N_FOLDS + 1), COLORS):
        sub = df[df['Fold'] == fold_id]
        ax.scatter(
            sub['Actual_FHR'], sub['Predicted_FHR'],
            color=color, label=f'Fold {fold_id}', s=60,
            edgecolor='white', linewidth=0.5, alpha=0.85,
        )

    all_vals = pd.concat([df['Actual_FHR'], df['Predicted_FHR']])
    lo, hi = all_vals.min() - 2, all_vals.max() + 2
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual FHR (bpm)', fontsize=12)
    ax.set_ylabel('Predicted FHR (bpm)', fontsize=12)
    ax.set_title('Predicted vs Actual FHR – All Folds', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fhr_cv_predictions_vs_actual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions vs actual plot saved → {path}")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'Records.csv')
    output_dir = base_dir  # save outputs next to the script

    # ── Load data ──
    X, y, df_labelled = load_and_preprocess(csv_path)

    # ── Cross-validation ──
    fold_results, all_predictions = run_cross_validation(X, y)

    # ── Display & save results ──
    display_results_table(fold_results)
    save_per_fold_csv(fold_results, os.path.join(output_dir, 'fhr_cv_results_per_fold.csv'))
    save_predictions_csv(all_predictions, os.path.join(output_dir, 'fhr_cv_predictions_all_folds.csv'))
    save_summary_report(fold_results, os.path.join(output_dir, 'fhr_cv_summary_report.txt'))

    # ── Visualisations ──
    plot_metrics_bar(fold_results, output_dir)
    plot_box_plots(fold_results, output_dir)
    plot_predictions_vs_actual(all_predictions, output_dir)

    print("\n✓ All done.")


if __name__ == '__main__':
    main()
