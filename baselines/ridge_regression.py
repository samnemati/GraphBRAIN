"""
ridge_regression.py — Ridge regression baseline on upper triangle of connectivity matrix.

Required by the critical 2025 paper (npj AI):
"Rethinking functional brain connectome analysis: do graph deep learning models Help?"
This baseline must be run and reported alongside GNN results.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from training.evaluate import compute_metrics
from config import N_FOLDS, SEED, N_REGIONS


def connectivity_to_vector(connectivity):
    """Flatten upper triangle of connectivity matrix to a feature vector."""
    idx = np.triu_indices(N_REGIONS, k=1)
    return connectivity[idx]


def run_ridge_baseline(subject_data, target_col, n_folds=N_FOLDS):
    """
    Run cross-validated ridge regression on flattened connectivity vectors.

    Returns:
        fold_metrics : list[dict]   per-fold metrics
        all_preds    : np.ndarray   concatenated validation predictions across folds
        all_targets  : np.ndarray   concatenated validation targets across folds
    """
    X = np.array([connectivity_to_vector(s["connectivity"]) for s in subject_data])
    y = np.array([s["behavioral"][target_col] for s in subject_data], dtype=float)

    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_preds = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        model.fit(X_train, y[train_idx])
        preds = model.predict(X_val)

        metrics = compute_metrics(y[val_idx], preds)
        fold_metrics.append(metrics)
        all_preds.append(preds)
        all_targets.append(y[val_idx])
        print(f"  Fold {fold+1}: MAE={metrics['mae']:.3f}  "
              f"R²={metrics['r2']:.3f}  r={metrics['pearson_r']:.3f}")

    return fold_metrics, np.concatenate(all_preds), np.concatenate(all_targets)
