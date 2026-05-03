"""
main.py - Entry point for the GraphMind pipeline.

Usage:
    python main.py --target MoCA_T --model gcn
    python main.py --target PTT_Average --model gcn
    python main.py --target MoCA_T --baseline_only
"""

import argparse
import json
import os
import torch

from config import (TARGET, MODEL_TYPE, NODE_FEATURES, RESULTS_DIR, SEED)
from data.dataset import build_dataset
from data.loader import load_all_subjects, load_behavioral_data
from models.gcn_regression import GCNRegression
from models.gat_regression import GATRegression
from training.train import cross_validate, set_seed
from training.evaluate import summarize_cv_results, plot_predictions
from baselines.ridge_regression import run_ridge_baseline


def _save_metrics_json(target: str, ridge_metrics, gcn_metrics, n_folds: int):
    """Write per-fold and summary metrics for both methods to a JSON file."""
    def _summarize(fm):
        if fm is None:
            return None
        return {
            "n_folds": len(fm),
            "per_fold": [{k: float(v) for k, v in m.items()} for m in fm],
            "mean": {k: float(sum(m[k] for m in fm) / len(fm))
                     for k in ["mae", "r2", "pearson_r"]},
        }
    out = {
        "target": target,
        "n_folds": n_folds,
        "node_features": list(NODE_FEATURES),
        "seed": SEED,
        "ridge": _summarize(ridge_metrics),
        "gcn": _summarize(gcn_metrics),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"metrics_{target.lower().replace(' ', '_')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


def _run_ridge(target: str):
    print(f"\n── Ridge Regression Baseline ({target}) ──")
    behavioral_df = load_behavioral_data()
    subject_data = load_all_subjects(behavioral_df=behavioral_df)
    metrics, preds, targets = run_ridge_baseline(subject_data, target_col=target)
    print("\nRidge Summary:")
    summarize_cv_results(metrics)
    return metrics, preds, targets


def _run_gcn(target: str, model_name: str, device):
    print(f"\n── {model_name.upper()} Cross-Validation ({target}) ──")
    dataset = build_dataset(target_col=target)
    model_class = GCNRegression if model_name == "gcn" else GATRegression
    model_kwargs = {"num_node_features": len(NODE_FEATURES)}
    fold_metrics, preds, targets = cross_validate(model_class, dataset,
                                                  model_kwargs, device)
    print(f"\n{model_name.upper()} Summary:")
    summarize_cv_results(fold_metrics)
    return fold_metrics, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--model", type=str, default=MODEL_TYPE,
                        choices=["gcn", "gat"])
    parser.add_argument("--baseline_only", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.target} | Model: {args.model} "
          f"| baseline_only={args.baseline_only}")

    target_slug = args.target.lower().replace(" ", "_")

    # Always run the ridge baseline; it's cheap and required by the spec.
    ridge_metrics, ridge_preds, ridge_targets = _run_ridge(args.target)

    # Ridge scatter plot
    ridge_path = os.path.join(RESULTS_DIR, f"scatter_{target_slug}_ridge.png")
    plot_predictions(ridge_targets, ridge_preds, target_name=f"{args.target} (ridge)",
                     save_path=ridge_path)
    print(f"Wrote {ridge_path}")

    if args.baseline_only:
        _save_metrics_json(args.target, ridge_metrics, None, len(ridge_metrics))
        return

    fold_metrics, gcn_preds, gcn_targets = _run_gcn(args.target, args.model, device)

    # GCN scatter plot
    gcn_path = os.path.join(RESULTS_DIR, f"scatter_{target_slug}_{args.model}.png")
    plot_predictions(gcn_targets, gcn_preds, target_name=f"{args.target} ({args.model})",
                     save_path=gcn_path)
    print(f"Wrote {gcn_path}")

    _save_metrics_json(args.target, ridge_metrics, fold_metrics, len(fold_metrics))


if __name__ == "__main__":
    main()
