"""
evaluate.py — Evaluation metrics and result visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred):
    """Return MAE, R², and Pearson r."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    return {"mae": mae, "r2": r2, "pearson_r": r, "pearson_p": p}


def summarize_cv_results(fold_metrics):
    """Print mean ± std across folds."""
    for metric in ["mae", "r2", "pearson_r"]:
        vals = [m[metric] for m in fold_metrics]
        print(f"{metric.upper():12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


def plot_predictions(y_true, y_pred, target_name="Target", save_path=None):
    """Scatter plot of predicted vs actual scores."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lim = [min(y_true.min(), y_pred.min()) - 1,
           max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"r = {r:.3f}  |  R² = {r2:.3f}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
