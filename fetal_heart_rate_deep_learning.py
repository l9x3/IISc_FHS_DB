"""
Comprehensive Deep Learning Pipeline for Fetal Heart Rate Prediction
=====================================================================
Dataset : IISc_features_complete_subset.csv
Task    : Multi-class classification – identify fetal subject from
          signal-processing features extracted from fetal heart sound recordings.
          Additionally, a regression head is evaluated for Heart_Rate prediction.

Models implemented:
  1. DNN  – Dense Neural Network (MLP)
  2. CNN  – 1-D Convolutional Neural Network
  3. LSTM – Long Short-Term Memory network
  4. Ensemble – soft-voting over DNN + CNN + LSTM predictions

Output files (written to ./results/):
  * training_history_<model>.png
  * confusion_matrix_<model>.png
  * feature_correlation_heatmap.png
  * roc_curves_<model>.png
  * classification_report_<model>.txt
  * saved_models/<model>.keras
  * model_comparison.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "IISc_features_complete_subset.csv"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "saved_models")
TARGET_COL = "Subject"
HEART_RATE_COL = "Heart_Rate"
TEST_SIZE = 0.2
EPOCHS = 150       # kept modest given the small dataset (66 samples)
BATCH_SIZE = 8     # small dataset – use small batch
CV_FOLDS = 5
LSTM_SEQ_LEN = 20  # number of time-steps used to reshape features for LSTM

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ===========================================================================
# 1. DATA LOADING & PREPROCESSING
# ===========================================================================

def load_and_preprocess(path: str):
    """Load CSV, drop NaNs, encode labels, scale features."""
    df = pd.read_csv(path)
    print(f"Dataset shape : {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Separate features / targets
    y_subject = df[TARGET_COL].values
    y_hr = df[HEART_RATE_COL].values if HEART_RATE_COL in df.columns else None
    feature_cols = [c for c in df.columns if c not in (TARGET_COL, HEART_RATE_COL)]
    X = df[feature_cols].values.astype(np.float32)

    # Encode class labels to 0-based integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y_subject)
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {le.classes_}")

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, y_hr, le, scaler, feature_cols, n_classes, df


def make_splits(X, y, test_size=TEST_SIZE):
    """Stratified train / test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )


# ===========================================================================
# 2. MODEL ARCHITECTURES
# ===========================================================================

def build_dnn(n_features: int, n_classes: int) -> keras.Model:
    """Deep Neural Network with batch normalisation and dropout."""
    inp = keras.Input(shape=(n_features,), name="dnn_input")
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="dnn_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="DNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(n_features: int, n_classes: int) -> keras.Model:
    """1-D CNN treating the feature vector as a 1-D signal."""
    inp = keras.Input(shape=(n_features, 1), name="cnn_input")

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="cnn_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(n_features: int, n_classes: int,
               seq_len: int = LSTM_SEQ_LEN) -> keras.Model:
    """
    LSTM treating the feature vector as a sequence of segments.
    The feature dimension is split into `seq_len` time-steps of length
    `n_features // seq_len` (padding if needed).
    """
    step = max(1, n_features // seq_len)
    actual_seq = n_features // step

    inp = keras.Input(shape=(actual_seq, step), name="lstm_input")

    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="lstm_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Add channel dimension for CNN input."""
    return X[..., np.newaxis]


def reshape_for_lstm(X: np.ndarray, seq_len: int = LSTM_SEQ_LEN) -> np.ndarray:
    """Reshape flat feature vector into (samples, seq_len, step)."""
    n_features = X.shape[1]
    step = max(1, n_features // seq_len)
    actual_seq = n_features // step
    # Truncate to exact multiple
    X_trunc = X[:, : actual_seq * step]
    return X_trunc.reshape(X_trunc.shape[0], actual_seq, step)


# ===========================================================================
# 3. CALLBACKS
# ===========================================================================

def make_callbacks(model_name: str, monitor: str = "val_accuracy"):
    ckpt_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    return [
        EarlyStopping(
            monitor=monitor,
            patience=20,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
    ]


# ===========================================================================
# 4. TRAINING
# ===========================================================================

def get_class_weights(y_train: np.ndarray, n_classes: int) -> dict:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train,
    )
    return {i: w for i, w in enumerate(weights)}


def train_model(model, X_tr, y_tr, X_val, y_val, model_name: str,
                n_classes: int):
    cw = get_class_weights(y_tr, n_classes)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=make_callbacks(model_name),
        verbose=0,
    )
    return history


# ===========================================================================
# 5. EVALUATION
# ===========================================================================

def evaluate_model(model, X_test, y_test, le: LabelEncoder, model_name: str):
    """Compute predictions and return metrics dict."""
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = float(np.mean(y_pred == y_test))
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    class_names = [str(c) for c in le.classes_]
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # Save classification report
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"=== {model_name} Classification Report ===\n\n")
        f.write(report)
    print(f"\n[{model_name}] Test accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}")
    print(report)

    return {
        "model": model_name,
        "accuracy": acc,
        "weighted_f1": f1,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
    }


# ===========================================================================
# 6. CROSS-VALIDATION
# ===========================================================================

def cross_validate_model(build_fn, X, y, n_classes: int,
                         model_name: str, seq_transform=None):
    """
    Run stratified k-fold CV and return mean / std accuracy and F1.
    `seq_transform` is an optional callable applied to X before feeding to model.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_accs, fold_f1s = [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if seq_transform is not None:
            X_tr_in = seq_transform(X_tr)
            X_val_in = seq_transform(X_val)
        else:
            X_tr_in, X_val_in = X_tr, X_val

        model = build_fn()
        cw = get_class_weights(y_tr, n_classes)
        model.fit(
            X_tr_in, y_tr,
            validation_data=(X_val_in, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=cw,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=20,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                  patience=10, min_lr=1e-6, verbose=0),
            ],
            verbose=0,
        )
        y_val_pred = np.argmax(model.predict(X_val_in, verbose=0), axis=1)
        acc = float(np.mean(y_val_pred == y_val))
        f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  [{model_name}] Fold {fold_idx}: acc={acc:.4f}  f1={f1:.4f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    print(f"  [{model_name}] CV  acc={mean_acc:.4f}±{std_acc:.4f}  "
          f"f1={mean_f1:.4f}±{std_f1:.4f}")
    return mean_acc, std_acc, mean_f1, std_f1


# ===========================================================================
# 7. ENSEMBLE
# ===========================================================================

def ensemble_predict(models_probs: list) -> np.ndarray:
    """Soft-voting ensemble: average predicted probabilities."""
    stacked = np.stack(models_probs, axis=0)  # (n_models, n_samples, n_classes)
    return stacked.mean(axis=0)


# ===========================================================================
# 8. VISUALISATION
# ===========================================================================

def plot_training_history(history, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} – Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{model_name} – Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"training_history_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, le: LabelEncoder, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(c) for c in le.classes_]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{model_name} – Confusion Matrix")
    ax.set_xlabel("Predicted Subject")
    ax.set_ylabel("True Subject")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_correlation(df: pd.DataFrame, feature_cols: list):
    """Plot correlation heatmap of the first 30 features for readability."""
    cols_to_plot = feature_cols[:30]
    corr = df[cols_to_plot].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr,
        annot=False,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap (first 30 features)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(y_true, y_pred_prob, le: LabelEncoder, model_name: str):
    """One-vs-rest ROC curves for each class."""
    n_classes = len(le.classes_)
    y_bin = to_categorical(y_true, num_classes=n_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab20", n_classes)

    auc_scores = []
    for i, cls in enumerate(le.classes_):
        try:
            auc = roc_auc_score(y_bin[:, i], y_pred_prob[:, i])
            auc_scores.append(auc)
            ax.plot([], [], color=colors(i),
                    label=f"Subject {cls} (AUC={auc:.2f})")
        except ValueError:
            pass  # only one class present in test set

    # Macro-average AUC
    try:
        macro_auc = roc_auc_score(y_bin, y_pred_prob,
                                  average="macro", multi_class="ovr")
        ax.set_title(
            f"{model_name} – ROC Curves (macro AUC = {macro_auc:.3f})"
        )
    except ValueError:
        ax.set_title(f"{model_name} – ROC Curves")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"roc_curves_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(comparison_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = comparison_df["Model"]

    axes[0].bar(models, comparison_df["Test Accuracy"], color="steelblue")
    axes[0].set_title("Model Comparison – Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comparison_df["Test Accuracy"]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(models, comparison_df["Weighted F1"], color="coral")
    axes[1].set_title("Model Comparison – Weighted F1")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(comparison_df["Weighted F1"]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# 9. FEATURE IMPORTANCE (permutation-based)
# ===========================================================================

def permutation_feature_importance(model, X_test, y_test,
                                   feature_cols: list, model_name: str,
                                   top_n: int = 20,
                                   input_transform=None):
    """
    Estimate feature importance by measuring accuracy drop when each feature
    is randomly shuffled.  Reports the top_n most important features.
    """
    if input_transform is not None:
        X_in = input_transform(X_test)
    else:
        X_in = X_test

    baseline_acc = float(np.mean(
        np.argmax(model.predict(X_in, verbose=0), axis=1) == y_test
    ))

    n_features = X_test.shape[1]
    importances = np.zeros(n_features)
    rng = np.random.RandomState(SEED)  # reproducible permutations

    for i in range(n_features):
        X_perm = X_test.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        if input_transform is not None:
            X_perm_in = input_transform(X_perm)
        else:
            X_perm_in = X_perm
        perm_acc = float(np.mean(
            np.argmax(model.predict(X_perm_in, verbose=0), axis=1) == y_test
        ))
        importances[i] = baseline_acc - perm_acc

    top_idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in top_idx]
    top_importances = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_importances[::-1], color="teal")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_xlabel("Mean accuracy decrease (permutation importance)")
    ax.set_title(f"{model_name} – Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return list(zip(top_features, top_importances))


# ===========================================================================
# 10. MAIN PIPELINE
# ===========================================================================

def main():
    print("=" * 70)
    print("Fetal Heart Rate Deep Learning Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    X, y, y_hr, le, scaler, feature_cols, n_classes, df = \
        load_and_preprocess(DATA_PATH)
    n_features = X.shape[1]

    print(f"\nFeatures      : {n_features}")
    print(f"Samples       : {X.shape[0]}")
    print(f"Classes       : {n_classes}")

    # Feature correlation heatmap (uses raw DataFrame)
    plot_feature_correlation(df, feature_cols)

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = make_splits(X, y)
    print(f"\nTrain samples : {X_train.shape[0]}")
    print(f"Test samples  : {X_test.shape[0]}")

    # Prepare reshaped copies for CNN / LSTM
    X_train_cnn = reshape_for_cnn(X_train)
    X_test_cnn = reshape_for_cnn(X_test)
    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)
    lstm_seq = X_train_lstm.shape[1]
    lstm_step = X_train_lstm.shape[2]

    comparison_rows = []

    # ------------------------------------------------------------------
    # 3. DNN
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Training DNN …")
    dnn = build_dnn(n_features, n_classes)
    dnn.summary()
    history_dnn = train_model(
        dnn, X_train, y_train, X_test, y_test, "DNN", n_classes
    )
    plot_training_history(history_dnn, "DNN")
    res_dnn = evaluate_model(dnn, X_test, y_test, le, "DNN")
    plot_confusion_matrix(y_test, res_dnn["y_pred"], le, "DNN")
    plot_roc_curves(y_test, res_dnn["y_pred_prob"], le, "DNN")
    permutation_feature_importance(dnn, X_test, y_test, feature_cols, "DNN")
    comparison_rows.append({
        "Model": "DNN",
        "Test Accuracy": res_dnn["accuracy"],
        "Weighted F1": res_dnn["weighted_f1"],
    })

    # ------------------------------------------------------------------
    # 4. CNN
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Training CNN …")
    cnn = build_cnn(n_features, n_classes)
    cnn.summary()
    history_cnn = train_model(
        cnn, X_train_cnn, y_train, X_test_cnn, y_test, "CNN", n_classes
    )
    plot_training_history(history_cnn, "CNN")
    res_cnn = evaluate_model(cnn, X_test_cnn, y_test, le, "CNN")
    plot_confusion_matrix(y_test, res_cnn["y_pred"], le, "CNN")
    plot_roc_curves(y_test, res_cnn["y_pred_prob"], le, "CNN")
    permutation_feature_importance(
        cnn, X_test, y_test, feature_cols, "CNN",
        input_transform=reshape_for_cnn
    )
    comparison_rows.append({
        "Model": "CNN",
        "Test Accuracy": res_cnn["accuracy"],
        "Weighted F1": res_cnn["weighted_f1"],
    })

    # ------------------------------------------------------------------
    # 5. LSTM
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Training LSTM …")
    lstm = build_lstm(n_features, n_classes)
    lstm.summary()
    history_lstm = train_model(
        lstm, X_train_lstm, y_train, X_test_lstm, y_test, "LSTM", n_classes
    )
    plot_training_history(history_lstm, "LSTM")
    res_lstm = evaluate_model(lstm, X_test_lstm, y_test, le, "LSTM")
    plot_confusion_matrix(y_test, res_lstm["y_pred"], le, "LSTM")
    plot_roc_curves(y_test, res_lstm["y_pred_prob"], le, "LSTM")
    permutation_feature_importance(
        lstm, X_test, y_test, feature_cols, "LSTM",
        input_transform=reshape_for_lstm
    )
    comparison_rows.append({
        "Model": "LSTM",
        "Test Accuracy": res_lstm["accuracy"],
        "Weighted F1": res_lstm["weighted_f1"],
    })

    # ------------------------------------------------------------------
    # 6. Ensemble (soft-voting)
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Evaluating Ensemble (DNN + CNN + LSTM soft-voting) …")
    ens_probs = ensemble_predict([
        res_dnn["y_pred_prob"],
        res_cnn["y_pred_prob"],
        res_lstm["y_pred_prob"],
    ])
    ens_pred = np.argmax(ens_probs, axis=1)
    ens_acc = float(np.mean(ens_pred == y_test))
    ens_f1 = f1_score(y_test, ens_pred, average="weighted", zero_division=0)
    class_names = [str(c) for c in le.classes_]
    ens_report = classification_report(
        y_test, ens_pred, target_names=class_names, zero_division=0
    )
    print(f"[Ensemble] Test accuracy: {ens_acc:.4f}  |  Weighted F1: {ens_f1:.4f}")
    print(ens_report)
    with open(os.path.join(RESULTS_DIR, "classification_report_Ensemble.txt"),
              "w") as f:
        f.write("=== Ensemble Classification Report ===\n\n")
        f.write(ens_report)
    plot_confusion_matrix(y_test, ens_pred, le, "Ensemble")
    plot_roc_curves(y_test, ens_probs, le, "Ensemble")
    comparison_rows.append({
        "Model": "Ensemble",
        "Test Accuracy": ens_acc,
        "Weighted F1": ens_f1,
    })

    # ------------------------------------------------------------------
    # 7. Cross-validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"Running {CV_FOLDS}-fold cross-validation …\n")

    cv_results = {}

    print("  DNN CV:")
    cv_results["DNN"] = cross_validate_model(
        lambda: build_dnn(n_features, n_classes),
        X, y, n_classes, "DNN"
    )

    print("  CNN CV:")
    cv_results["CNN"] = cross_validate_model(
        lambda: build_cnn(n_features, n_classes),
        X, y, n_classes, "CNN",
        seq_transform=reshape_for_cnn,
    )

    print("  LSTM CV:")
    cv_results["LSTM"] = cross_validate_model(
        lambda: build_lstm(n_features, n_classes, seq_len=LSTM_SEQ_LEN),
        X, y, n_classes, "LSTM",
        seq_transform=reshape_for_lstm,
    )

    # Append CV results to comparison table
    for name, (m_acc, s_acc, m_f1, s_f1) in cv_results.items():
        for row in comparison_rows:
            if row["Model"] == name:
                row["CV Accuracy (mean±std)"] = f"{m_acc:.4f}±{s_acc:.4f}"
                row["CV F1 (mean±std)"] = f"{m_f1:.4f}±{s_f1:.4f}"

    # ------------------------------------------------------------------
    # 8. Model comparison
    # ------------------------------------------------------------------
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nModel comparison saved to {csv_path}")
    print(comparison_df.to_string(index=False))

    # Visualise comparison (only numeric accuracy / F1 columns)
    plot_model_comparison(comparison_df[["Model", "Test Accuracy",
                                        "Weighted F1"]])

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pipeline complete. All outputs saved in:", RESULTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
