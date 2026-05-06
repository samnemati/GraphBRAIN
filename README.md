# GraphMind

Graph Neural Network regression pipeline for predicting behavioral outcomes
(cognitive and hearing scores) from resting-state functional brain connectivity
in the ABC Healthy Aging cohort (ages 20–80, N≈305).

## Overview

Each subject is represented as a graph: 189 JHU-atlas regions are nodes,
top-15% of resting-state functional connectivity (`rest_jhu['r']`) provides the
edges, and per-region FA + MD form the node features. PyTorch Geometric models
(`GraphConv`-based GCN; GAT to come) regress directly to a single behavioral
score. A ridge regression on the flattened upper triangle of the connectivity
matrix runs as a baseline on every call.

Targets:
- **MoCA_T** — Montreal Cognitive Assessment score
- **PTT_Average** — Pure Tone Threshold (hearing)
- *(planned)* WIN_Threshold_Average, Age sanity check

## Quickstart

```bash
# Single-subject end-to-end smoke test (gate before any training run)
python scripts/smoke_test.py

# Ridge baseline + GCN cross-validation, with scatter plots and metrics JSON
python main.py --target MoCA_T --model gcn
python main.py --target PTT_Average --model gcn

# Ridge only (fast — skips GCN training)
python main.py --target MoCA_T --baseline_only

# 12-config hyperparameter sweep on PTT_Average GCN (Phase 2 of tuning sprint)
python scripts/sweep_ptt_gcn.py
```

Outputs are written to `results/`:
- `scatter_<target>_{ridge,gcn}.png` — concatenated cross-fold predictions
- `metrics_<target>.json` — per-fold + mean MAE / R² / Pearson r
- `cache/dataset_<target>_<hash>.pkl` — built-graphs cache (re-runs in seconds)

The dataset cache key hashes `target`, `NODE_FEATURES`, `THRESHOLD_PERCENT`,
`ATLAS`, and `EDGE_TYPE` — change any of these in `config.py` and the cache is
rebuilt automatically. Runs are deterministic from `SEED=42`.

## Current Results

5-fold cross-validation, mean across folds. From
`results/metrics_moca_t.json` and `results/metrics_ptt_average.json`.

| Target | Method | MAE | R² | Pearson r |
|---|---|---|---|---|
| MoCA_T | Ridge | 1.981 | -0.087 | 0.247 |
| MoCA_T | GCN | 2.333 | -0.422 | 0.106 |
| PTT_Average | Ridge | 8.796 | 0.267 | 0.543 |
| PTT_Average | GCN | 8.441 | 0.219 | 0.509 |

Ridge currently outperforms the GCN on R² for both targets — directly
consistent with the 2025 npj AI "do graph deep learning models help?"
finding. The PTT GCN tuning sprint is the active workstream to close that
gap.

## Project Structure

```
GraphMind/
├── main.py                    ← entry point: ridge + GCN cross-validation driver
├── config.py                  ← paths, hyperparameters, seeds
├── data/
│   ├── loader.py              ← scipy.io.loadmat → connectivity + per-region features
│   ├── graph_builder.py       ← connectivity matrix → PyG Data object
│   └── dataset.py             ← build_dataset() with disk caching
├── models/
│   ├── gcn_regression.py      ← GraphConv-based GCN with signed edge weights
│   └── gat_regression.py      ← GAT variant (for interpretability sprint)
├── training/
│   ├── train.py               ← cross_validate() with per-fold standardization + early stopping
│   └── evaluate.py            ← MAE / R² / Pearson r metrics + scatter plot
├── baselines/
│   └── ridge_regression.py    ← RidgeCV on flattened upper-triangle connectivity
├── interpretation/
│   └── explain.py             ← GNNExplainer + attention weights (planned)
├── scripts/
│   ├── inspect_mat.py         ← one-shot .mat structure inspector
│   ├── smoke_test.py          ← single-subject end-to-end check
│   └── sweep_ptt_gcn.py       ← 12-config hyperparameter sweep
└── results/                   ← scatter PNGs, metrics JSON, dataset cache (gitignored)
```

## Sprint History

| Date | Sprint | Status |
|---|---|---|
| 2026-04-30 | First end-to-end run: wire up the full pipeline, ridge baseline + GCN CV on MoCA + PTT, seeded reproducibility, dataset caching | Done |
| 2026-05-02 | PTT GCN tuning: switch `GCNConv` to `GraphConv` for signed edge weights (Phase 1); 12-config `lr × dropout × num_layers` sweep with early stopping (Phase 2) | In progress |

Planned next: GAT + interpretability sprint (attention weights, GNNExplainer
on the better-performing target), then multi-task / WIN_Threshold extension.

## Setup

PyTorch and PyTorch Geometric must be installed first, matching your platform —
see the [PyTorch](https://pytorch.org/get-started/locally/) and
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
install guides. Then:

```bash
pip install -r requirements.txt
```

Data paths in `config.py` are absolute and point at the local ABC dataset
directory; adjust `DATA_ROOT` if running elsewhere.
