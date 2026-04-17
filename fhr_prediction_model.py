"""
Fetal Heart Rate (FHR) Prediction Deep Learning Model
======================================================
Predicts FHR from clinical tabular features using a neural network.
Dataset: Records.csv (60 pregnant women, 44 with FHR measurements)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
FHR_MIN = 110.0
FHR_MAX = 160.0
MAX_EPOCHS = 300
BATCH_SIZE = 4   # small dataset (~30 train samples) – use small batch for gradient stability
FEATURES = [
    'gestational_age',
    'maternal_age',
    'weight',
    'height',
    'gravida',
    'para',
    'bmi',
    'clinical_risk_score'
]

# Output directory for plots
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> tuple:
    """Load Records.csv, drop rows without FHR, return X and y."""
    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print("  Fetal Heart Rate Prediction – Data Summary")
    print(f"{'='*55}")
    print(f"Total records loaded       : {len(df)}")

    # Keep only rows with FHR measurement
    df_fhr = df.dropna(subset=['fhr']).reset_index(drop=True)
    print(f"Records with FHR values    : {len(df_fhr)}")
    print(f"FHR range in dataset       : "
          f"{df_fhr['fhr'].min():.1f} – {df_fhr['fhr'].max():.1f} bpm")
    print(f"\nFeatures used ({len(FEATURES)}):")
    for feat in FEATURES:
        print(f"  • {feat}")

    X = df_fhr[FEATURES].values.astype(np.float32)
    y = df_fhr['fhr'].values.astype(np.float32)
    return X, y


def split_and_scale(X: np.ndarray, y: np.ndarray) -> tuple:
    """70% train / 15% val / 15% test split with StandardScaler."""
    # First split off 30% for val+test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    # Split the 30% equally → each 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\nDataset split:")
    print(f"  Train : {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Val   : {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  Test  : {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────────
# 2. Model Architecture
# ─────────────────────────────────────────────
def build_model(n_features: int) -> keras.Model:
    """
    Neural network for FHR regression.
    - Input  : 8 clinical features
    - Hidden : 3 dense layers with BatchNorm + Dropout
    - Output : sigmoid activation scaled to [110, 160] bpm
    """
    inputs = keras.Input(shape=(n_features,), name='clinical_features')

    # Hidden layer 1
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4),
                     name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)

    # Hidden layer 2
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4),
                     name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)

    # Hidden layer 3
    x = layers.Dense(16, kernel_regularizer=regularizers.l2(1e-4),
                     name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(0.1, name='dropout_3')(x)

    # Output: sigmoid → scale to [FHR_MIN, FHR_MAX]
    raw_out = layers.Dense(1, activation='sigmoid', name='output_sigmoid')(x)
    outputs = layers.Lambda(
        lambda t: t * (FHR_MAX - FHR_MIN) + FHR_MIN,
        name='fhr_bpm'
    )(raw_out)

    model = keras.Model(inputs=inputs, outputs=outputs, name='FHR_Predictor')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


# ─────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────
def train_model(model: keras.Model,
                X_train, y_train,
                X_val, y_val) -> keras.callbacks.History:
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    print(f"\nTraining …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )
    print(f"Training finished after {len(history.history['loss'])} epochs.")
    return history


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1e-9))) * 100

    metrics = dict(mae=mae, rmse=rmse, r2=r2, mape=mape,
                   y_pred=y_pred, y_test=y_test)

    print(f"\n{'='*55}")
    print("  Test-Set Evaluation Metrics")
    print(f"{'='*55}")
    print(f"  MAE   (Mean Absolute Error)         : {mae:.4f} bpm")
    print(f"  RMSE  (Root Mean Squared Error)     : {rmse:.4f} bpm")
    print(f"  R²    Score                         : {r2:.4f}")
    print(f"  MAPE  (Mean Abs. Percentage Error)  : {mape:.4f} %")
    print(f"{'='*55}\n")
    return metrics


# ─────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────
def plot_training_history(history: keras.callbacks.History,
                          save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='tomato')
    axes[0].set_title('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE (bpm²)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', color='steelblue')
    axes[1].plot(history.history['val_mae'], label='Val MAE', color='tomato')
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (bpm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_predictions_vs_actual(y_test: np.ndarray,
                               y_pred: np.ndarray,
                               save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color='steelblue', edgecolors='white',
               alpha=0.8, s=60, label='Predictions')

    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual FHR (bpm)', fontsize=12)
    ax.set_ylabel('Predicted FHR (bpm)', fontsize=12)
    ax.set_title('Predictions vs Actual FHR', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_residuals(y_test: np.ndarray,
                   y_pred: np.ndarray,
                   save_path: str) -> None:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, color='steelblue', edgecolors='white',
               alpha=0.8, s=60)
    ax.axhline(0, color='tomato', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Predicted FHR (bpm)', fontsize=12)
    ax.set_ylabel('Residual (Actual − Predicted) (bpm)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_error_distribution(y_test: np.ndarray,
                            y_pred: np.ndarray,
                            save_path: str) -> None:
    errors = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=10, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='tomato', linewidth=1.5, linestyle='--', label='Zero error')
    ax.axvline(errors.mean(), color='gold', linewidth=1.5, linestyle='-.',
               label=f'Mean error: {errors.mean():.2f} bpm')
    ax.set_xlabel('Prediction Error (bpm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    csv_path = 'Records.csv'
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. "
            "Please place Records.csv in the current working directory."
        )

    # Load & preprocess
    X, y = load_data(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(X, y)

    # Build model
    model = build_model(n_features=len(FEATURES))
    model.summary()

    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'fhr_model_final.keras')
    model.save(model_path)
    print(f"  Model saved → {model_path}")

    # Visualisations
    print("\nGenerating visualisations …")
    plot_training_history(
        history,
        os.path.join(OUTPUT_DIR, 'training_history.png')
    )
    plot_predictions_vs_actual(
        metrics['y_test'], metrics['y_pred'],
        os.path.join(OUTPUT_DIR, 'predictions_vs_actual.png')
    )
    plot_residuals(
        metrics['y_test'], metrics['y_pred'],
        os.path.join(OUTPUT_DIR, 'residual_plot.png')
    )
    plot_error_distribution(
        metrics['y_test'], metrics['y_pred'],
        os.path.join(OUTPUT_DIR, 'error_distribution.png')
    )

    print("\nAll done! Outputs saved in ./outputs/")
    return metrics


if __name__ == '__main__':
    main()
