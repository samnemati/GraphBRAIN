"""
sweep_ptt_gcn.py - 12-config hyperparameter sweep for PTT_Average GCN.

Grid: lr × dropout × num_layers = 3 × 2 × 2 = 12 configs.
Other hyperparameters fixed at config.py defaults (HIDDEN_DIM, WEIGHT_DECAY,
BATCH_SIZE, SEED, EPOCHS hard cap; early stopping is in cross_validate).

Output: results/metrics_ptt_average_sweep.json with one block per config plus
a top-level 'best' field. Then applies the per-fold dominance rule to the
top-ranked-by-mean-R² config.
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NODE_FEATURES, RESULTS_DIR
from data.dataset import build_dataset
from models.gcn_regression import GCNRegression
from training.train import cross_validate, set_seed

TARGET = "PTT_Average"
RIDGE_R2 = [0.4244869466339638, 0.24571375290580022, 0.3291454691681597,
            0.23107839766916916, 0.10217144297833158]
RIDGE_MEAN_R2 = sum(RIDGE_R2) / len(RIDGE_R2)

LR_GRID = [1e-4, 5e-4, 1e-3]
DROPOUT_GRID = [0.2, 0.5]
LAYERS_GRID = [2, 3]


def _summarize(fold_metrics):
    return {
        "n_folds": len(fold_metrics),
        "per_fold": [{k: float(v) for k, v in m.items()} for m in fold_metrics],
        "mean": {k: float(sum(m[k] for m in fold_metrics) / len(fold_metrics))
                 for k in ["mae", "r2", "pearson_r"]},
    }


def _per_fold_dominance(per_fold_r2, ridge_r2):
    wins = sum(1 for g, r in zip(per_fold_r2, ridge_r2) if g > r)
    return wins, [g > r for g, r in zip(per_fold_r2, ridge_r2)]


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {TARGET}  | grid: {len(LR_GRID)*len(DROPOUT_GRID)*len(LAYERS_GRID)} configs")

    dataset = build_dataset(target_col=TARGET)
    n_features = len(NODE_FEATURES)

    configs = []
    for lr in LR_GRID:
        for dropout in DROPOUT_GRID:
            for num_layers in LAYERS_GRID:
                configs.append({"lr": lr, "dropout": dropout, "num_layers": num_layers})

    results = []
    for i, cfg in enumerate(configs, 1):
        tag = f"lr={cfg['lr']:.0e} dropout={cfg['dropout']} layers={cfg['num_layers']}"
        print(f"\n══════ Config {i}/{len(configs)}: {tag} ══════")
        model_kwargs = {
            "num_node_features": n_features,
            "dropout": cfg["dropout"],
            "num_layers": cfg["num_layers"],
        }
        fold_metrics, _, _ = cross_validate(
            GCNRegression, dataset, model_kwargs, device, lr=cfg["lr"]
        )
        summary = _summarize(fold_metrics)
        wins, fold_wins = _per_fold_dominance(
            [m["r2"] for m in fold_metrics], RIDGE_R2
        )
        results.append({
            "config": cfg,
            "metrics": summary,
            "ridge_per_fold_dominance": {
                "wins": wins,
                "fold_wins": fold_wins,
                "ridge_per_fold_r2": RIDGE_R2,
            },
        })
        print(f"  Mean R²={summary['mean']['r2']:.3f}  "
              f"per-fold ridge wins={wins}/5")

    # Rank configs by mean R²
    results_sorted = sorted(results, key=lambda r: r["metrics"]["mean"]["r2"], reverse=True)
    best = results_sorted[0]

    # Apply per-fold dominance to the top-ranked candidate
    best_wins = best["ridge_per_fold_dominance"]["wins"]
    sprint_success = best_wins >= 3

    out = {
        "target": TARGET,
        "ridge_per_fold_r2": RIDGE_R2,
        "ridge_mean_r2": RIDGE_MEAN_R2,
        "n_configs": len(configs),
        "configs_ranked_by_mean_r2": results_sorted,
        "best_config": best["config"],
        "best_mean_r2": best["metrics"]["mean"]["r2"],
        "best_per_fold_wins_vs_ridge": best_wins,
        "sprint_success": sprint_success,
    }
    out_path = os.path.join(RESULTS_DIR, "metrics_ptt_average_sweep.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n══════ Sweep summary ══════")
    print(f"Best config: {best['config']}")
    print(f"Best mean R²: {best['metrics']['mean']['r2']:.3f} "
          f"(ridge mean R²: {RIDGE_MEAN_R2:.3f})")
    print(f"Per-fold wins vs ridge: {best_wins}/5")
    print(f"Sprint success: {sprint_success}")


if __name__ == "__main__":
    main()
