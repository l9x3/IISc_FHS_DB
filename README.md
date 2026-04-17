# IISc_FHS_DB
Fetal Heart Sound Dataset

## Deep Learning Pipeline

`fetal_heart_rate_deep_learning.py` implements a comprehensive deep learning
pipeline for fetal subject identification from heart-sound signal features.

### Dataset

| Property | Value |
|---|---|
| File | `IISc_features_complete_subset.csv` |
| Samples | 66 |
| Features | 212 signal-processing features |
| Target | Subject ID (13 classes: 4–18) |
| Also available | `Heart_Rate` column |

> **Note on dataset size:** With only 66 samples and 212 features across 13
> classes (≈5 samples/class), deep learning models will achieve modest accuracy
> due to extreme class imbalance and limited data.  Results should be
> interpreted as a proof-of-concept.  Consider collecting more recordings or
> applying data-augmentation (e.g. Gaussian noise, time-warping on the raw
> audio) before drawing clinical conclusions.

### Models

| Model | Description |
|---|---|
| **DNN** | Dense Neural Network with batch normalisation and dropout |
| **CNN** | 1-D Convolutional Network treating features as a 1-D signal |
| **LSTM** | LSTM network treating features as a temporal sequence |
| **Ensemble** | Soft-voting combination of all three models |

### Requirements

```
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Usage

```bash
python fetal_heart_rate_deep_learning.py
```

All outputs are written to the `results/` directory:

```
results/
├── training_history_<DNN|CNN|LSTM>.png
├── confusion_matrix_<DNN|CNN|LSTM|Ensemble>.png
├── roc_curves_<DNN|CNN|LSTM|Ensemble>.png
├── feature_importance_<DNN|CNN|LSTM>.png
├── feature_correlation_heatmap.png
├── classification_report_<DNN|CNN|LSTM|Ensemble>.txt
├── model_comparison.csv
├── model_comparison.png
└── saved_models/
    ├── DNN.keras
    ├── CNN.keras
    └── LSTM.keras
```

### Pipeline Steps

1. **Data loading & preprocessing** – standardise features, encode labels, stratified split
2. **Model training** – early stopping, reduce-LR-on-plateau, model checkpointing, class-weight balancing
3. **Evaluation** – accuracy, weighted F1, classification report, confusion matrix, ROC curves
4. **Cross-validation** – 5-fold stratified CV for DNN, CNN, and LSTM
5. **Ensemble** – soft-voting over DNN + CNN + LSTM predictions
6. **Feature importance** – permutation-based importance for each model
7. **Visualisation** – training history, confusion matrices, ROC curves, feature correlation heatmap, model comparison bar chart
