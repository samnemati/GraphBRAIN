# PTT GCN Tuning Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the GCN to beat the ridge baseline on `PTT_Average` per-fold (≥ 3 of 5 folds), first by passing signed edge weights through the conv layer (Phase 1), then via a 12-config hyperparameter sweep with early stopping if Phase 1 fails (Phase 2).

**Architecture:** Phase 1 swaps `GCNConv` (which requires non-negative weights) for `GraphConv` (a weighted-sum conv that handles signed weights natively), and threads `edge_weight=data.edge_attr.squeeze(-1)` through `forward()`. If that doesn't clear the per-fold-dominance bar, Phase 2 adds early stopping to `cross_validate` and runs a sweep over `lr × dropout × num_layers` (3 × 2 × 2 = 12) via a new `scripts/sweep_ptt_gcn.py`.

**Tech Stack:** PyTorch, PyTorch Geometric (`GraphConv`, `global_mean_pool`), scikit-learn (KFold), JSON for sweep results.

**Spec:** [docs/superpowers/specs/2026-05-02-ptt-gcn-tuning-design.md](../specs/2026-05-02-ptt-gcn-tuning-design.md)

**Ridge baseline to beat (per-fold R², from prior sprint):**
`[0.4244869466339638, 0.24571375290580022, 0.3291454691681597, 0.23107839766916916, 0.10217144297833158]`
Decision rule: GCN per-fold R² must exceed ridge R² on **≥ 3 of 5 folds**.

---

## File Structure

```
GraphMind/
├── models/
│   └── gcn_regression.py                 [Task 1 — GCNConv → GraphConv, signed edge_weight]
├── training/
│   └── train.py                          [Task 4 — early stopping + lr param (Phase 2 only)]
├── scripts/
│   └── sweep_ptt_gcn.py                  [Task 5 — NEW driver for 12-config sweep]
└── results/
    ├── scatter_ptt_average_gcn.png       [updated by Task 2 (Phase 1 re-run)]
    ├── metrics_ptt_average.json          [updated by Task 2]
    └── metrics_ptt_average_sweep.json    [Task 5 output, gitignored]
```

Tasks are sequenced so Phase 2 work (Tasks 4 + 5) only happens if Task 3's evaluation says Phase 1 didn't clear the bar.

---

## Task 1: Replace `GCNConv` with `GraphConv` and pass signed edge weights

**Files:**
- Modify: `models/gcn_regression.py` (full rewrite — short file)

**Decision rule for the implementer:** the existing module has class `GCNRegression` and uses `GCNConv` from `torch_geometric.nn`. We are NOT renaming the class or file (call sites in `main.py` and `scripts/smoke_test.py` rely on the name). We're replacing the conv layer internally and threading `edge_weight` through.

- [ ] **Step 1: Replace `models/gcn_regression.py` end-to-end**

Overwrite the whole file with:

```python
"""
gcn_regression.py - Graph regression network with signed-weight message passing.

Why GraphConv (not GCNConv):
- GCNConv applies symmetric normalization (D^(-1/2) · A · D^(-1/2)) which
  mathematically requires non-negative edge weights. The Pearson correlations
  in data.edge_attr are signed (typically -0.3 to +0.9 in our top-15% edges).
- GraphConv computes x_i' = W1·x_i + W2·sum_j(e_ij · x_j): a weighted sum of
  neighbour features with a separate self-transform, no symmetric normalization.
  Signed weights are mathematically valid here — positive edges add the
  neighbour's features, negative edges subtract them, scaled by magnitude.

Architecture:
    GraphConv layers (signed-weight message passing) → BatchNorm → ReLU → Dropout
    → Global mean pool → Linear regression head (scalar output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from config import HIDDEN_DIM, NUM_LAYERS, DROPOUT


class GCNRegression(nn.Module):
    def __init__(self, num_node_features, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GraphConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is shape [num_edges, 1]; GraphConv wants edge_weight as [num_edges]
        edge_weight = edge_attr.squeeze(-1)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)   # graph-level representation
        return self.head(x).squeeze(-1)  # scalar prediction per graph
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python -c "from models.gcn_regression import GCNRegression; print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Verify the smoke test still passes**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python scripts/smoke_test.py
```
Expected: prints `SMOKE TEST PASSED` as the last line. The pred value will differ from before (~0.07 with the old GCNConv) — that's expected; what matters is that no shapes break and no NaN/Inf appears (the smoke test asserts both).

If shape mismatches appear: confirm `edge_attr.squeeze(-1)` produces a 1-D tensor of length `num_edges`. If the script crashes with a CUDA-vs-CPU dtype error, ensure `edge_weight` is `float32` (it should be — `edge_attr` is created as `torch.float` in `data/graph_builder.py:68`).

- [ ] **Step 4: Commit**

```bash
git add models/gcn_regression.py
git commit -m "models: switch GCN to GraphConv for signed edge weights

Pearson correlations in data.edge_attr are signed (typically -0.3 to +0.9
in the surviving top-15% edges). GCNConv's symmetric normalization
(D^(-1/2) A D^(-1/2)) requires non-negative weights, so we couldn't pass
edge_attr through it without abs() — which would erase a biologically
meaningful sign distinction.

GraphConv computes W1*x + W2*sum_j(e_ij*x_j): weighted sum, no symmetric
normalization, signed weights are mathematically valid."
```

---

## Task 2: Re-run PTT_Average GCN with signed edge weights

**Files:** none (verification step). Updates `results/scatter_ptt_average_gcn.png` and the `gcn` block of `results/metrics_ptt_average.json`.

- [ ] **Step 1: Run the GCN cross-validation**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python main.py --target PTT_Average --model gcn 2>/tmp/ptt_gcn_signed_stderr.log >/tmp/ptt_gcn_signed_stdout.log
```

Run this in the foreground or background — your choice. The dataset cache is reused (no .mat reloading), so total time is just GCN training (~10–25 min on CPU). Ridge runs first as part of `main.py` and adds ~3 min for the data load.

- [ ] **Step 2: Verify clean exit and per-fold metrics produced**

Run:
```bash
echo "exit=$?" && grep -E "Final →|GCN Summary:|MAE +:|R2 +:|PEARSON|Wrote" /tmp/ptt_gcn_signed_stdout.log | tail -25
```

Expected: `exit=0`, 5 lines beginning `Final →`, the GCN Summary block, and the two `Wrote …` lines for the scatter PNG and the metrics JSON.

If `Traceback` or `Killed` appears in `/tmp/ptt_gcn_signed_stderr.log`: do NOT proceed. Report the error and stop. Likely causes (in priority order):
- BatchNorm1d failing on a single-graph batch — would only happen with batch_size=1 in some folds; not expected with our default 16, but flag if seen.
- Shape mismatch in `edge_weight` — re-check Task 1, Step 1.

- [ ] **Step 3: Inspect the metrics JSON**

Run:
```bash
cat results/metrics_ptt_average.json | python3 -m json.tool | grep -A 30 '"gcn"'
```

Expected: 5 per-fold entries each with `mae`, `r2`, `pearson_r`, `pearson_p`. The `mean` block at the end has `mae`, `r2`, `pearson_r` keys.

- [ ] **Step 4: No commit (artifacts gitignored)**

The scatter PNG and the metrics JSON are gitignored per the previous sprint's `.gitignore` rules. Move on to Task 3.

---

## Task 3: Evaluate Phase 1 against the per-fold dominance rule

**Files:** none (decision step).

- [ ] **Step 1: Compute per-fold dominance**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python -c "
import json
RIDGE_R2 = [0.4244869466339638, 0.24571375290580022, 0.3291454691681597,
            0.23107839766916916, 0.10217144297833158]
d = json.load(open('results/metrics_ptt_average.json'))
gcn_r2 = [f['r2'] for f in d['gcn']['per_fold']]
wins = sum(1 for g, r in zip(gcn_r2, RIDGE_R2) if g > r)
print(f'GCN R² per fold:   {[f\"{x:.3f}\" for x in gcn_r2]}')
print(f'Ridge R² per fold: {[f\"{x:.3f}\" for x in RIDGE_R2]}')
print(f'Wins by fold:      {[g > r for g, r in zip(gcn_r2, RIDGE_R2)]}')
print(f'GCN wins {wins}/5 folds. Decision threshold: >=3.')
print(f'PHASE 1 SUCCESS' if wins >= 3 else f'PHASE 1 FAIL — proceed to Phase 2 (Tasks 4 + 5).')
"
```

- [ ] **Step 2: Branch on the result**

- If the script prints `PHASE 1 SUCCESS` → **stop here**, sprint complete. Skip Tasks 4 and 5. Report the result and move on.
- If the script prints `PHASE 1 FAIL` → continue to Task 4.

---

## Task 4: Add early stopping + `lr` parameter to `cross_validate(...)`

**Files:**
- Modify: `training/train.py` (single function, replace body)

**Implementation note:** the existing `cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS)` returns `(fold_metrics, all_preds, all_targets)` and runs all `EPOCHS` (200) epochs every fold, reporting the final-epoch metrics. We add:
1. An optional `lr` keyword argument (default `LEARNING_RATE` from config).
2. Early stopping with patience=20 on validation MAE, restoring the best-MAE weights before final evaluation.

The existing `main.py` call site (`cross_validate(model_class, dataset, model_kwargs, device)`) keeps working because `lr` is keyword-only with a default.

- [ ] **Step 1: Replace the `cross_validate` function body**

Find the existing `def cross_validate(...)` in `training/train.py` and replace the entire function with:

```python
def cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS,
                   lr=None, patience=20):
    """Run k-fold cross-validation with per-fold node-feature standardization
    and early stopping on validation MAE.

    Args:
        lr: optional learning rate; falls back to config.LEARNING_RATE.
        patience: epochs without val-MAE improvement before stopping.

    Returns: (fold_metrics, all_preds, all_targets)
    """
    set_seed(SEED)
    if lr is None:
        lr = LEARNING_RATE

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    indices = list(range(len(dataset)))
    fold_metrics = []
    all_preds = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n── Fold {fold + 1}/{n_folds} ──")
        train_data_raw = [dataset[i] for i in train_idx]
        val_data_raw = [dataset[i] for i in val_idx]
        train_data, val_data = standardize_node_features(train_data_raw, val_data_raw)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr, weight_decay=WEIGHT_DECAY)

        best_mae = float("inf")
        best_state = None
        best_epoch = 0
        epochs_since_best = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            preds, targets = evaluate(model, val_loader, device)
            metrics = compute_metrics(targets, preds)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
                      f"r: {metrics['pearson_r']:.3f}")

            if epochs_since_best >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"best val MAE {best_mae:.3f}).")
                break

        # Restore best-MAE weights for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)

        preds, targets = evaluate(model, val_loader, device)
        metrics = compute_metrics(targets, preds)
        fold_metrics.append(metrics)
        all_preds.append(preds)
        all_targets.append(targets)
        print(f"  Final (best-weights) → MAE: {metrics['mae']:.3f} | "
              f"R²: {metrics['r2']:.3f} | r: {metrics['pearson_r']:.3f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return fold_metrics, all_preds, all_targets
```

- [ ] **Step 2: Verify the import path still works**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python -c "from training.train import cross_validate; print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Verify backward compatibility — main.py still drives correctly**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses')"
```
Expected: prints `main.py parses`. (We're not running main.py here — that would re-train; we just confirm the source is still syntactically valid.)

- [ ] **Step 4: Commit**

```bash
git add training/train.py
git commit -m "training: add early stopping + optional lr parameter to cross_validate

- Patience: 20 epochs on validation MAE, restore best-MAE weights before
  final evaluation (Final line label updated to '(best-weights)' to make
  this explicit).
- New keyword-only lr param defaults to config.LEARNING_RATE; lets the
  Phase-2 sweep override per-config without touching config.py.
- Hard cap stays at EPOCHS (200); patience-driven stop usually triggers
  much earlier.
- main.py call site is unchanged (uses defaults)."
```

---

## Task 5: Build `scripts/sweep_ptt_gcn.py` and run the 12-config sweep

**Files:**
- Create: `scripts/sweep_ptt_gcn.py`

**Sweep grid:** `lr ∈ {1e-4, 5e-4, 1e-3}` × `dropout ∈ {0.2, 0.5}` × `num_layers ∈ {2, 3}` = 12 configs.

The driver runs the cross-validation 12 times, collects fold-level metrics for each config, ranks configs by mean R², and applies the per-fold dominance rule to the top-ranked config.

- [ ] **Step 1: Create `scripts/sweep_ptt_gcn.py`**

```python
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
```

- [ ] **Step 2: Run the sweep**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && KMP_DUPLICATE_LIB_OK=TRUE /Users/snemati/miniconda3/envs/pybrainage_env2/bin/python scripts/sweep_ptt_gcn.py 2>/tmp/sweep_stderr.log >/tmp/sweep_stdout.log
```

Run in the background if you don't want to block. Estimated runtime: 60–90 minutes on CPU (12 configs × 5 folds × ~80 epochs after early stopping × ~0.9 sec/epoch). The dataset is cached so no .mat reload.

If runtime exceeds 2 hours: stop, investigate why early stopping isn't triggering. Common causes:
- Patience too high — try 10 instead of 20 (would require Task 4 edit + re-run).
- Validation MAE oscillating without trending — model may be too noisy at high lr; lower-lr configs should still finish.

- [ ] **Step 3: Verify clean exit and inspect output**

Run:
```bash
echo "exit=$?" && grep -E "Config |Mean R²|Sprint success|Wrote" /tmp/sweep_stdout.log | tail -30
```

Expected: 12 `Config N/12:` lines and 12 `Mean R²=...` lines, ending with `Sprint success: True` or `Sprint success: False` and one `Wrote` line for the JSON output.

If `Traceback`: do NOT proceed. Diagnose by reading `/tmp/sweep_stderr.log`. Most likely failure is an `lr` plumbing issue from Task 4 — verify `cross_validate(... , lr=...)` was applied correctly.

- [ ] **Step 4: Inspect the sweep JSON**

Run:
```bash
cat results/metrics_ptt_average_sweep.json | python3 -m json.tool | head -80
```

Expected: top-level fields `target`, `ridge_per_fold_r2`, `n_configs=12`, `configs_ranked_by_mean_r2` (a 12-element list), `best_config`, `best_mean_r2`, `best_per_fold_wins_vs_ridge`, `sprint_success`.

- [ ] **Step 5: Commit the sweep script (output file is gitignored)**

```bash
git add scripts/sweep_ptt_gcn.py
git commit -m "scripts: 12-config sweep driver for PTT_Average GCN tuning

Grid: lr in {1e-4, 5e-4, 1e-3} x dropout in {0.2, 0.5} x num_layers in {2, 3}.
Output: results/metrics_ptt_average_sweep.json (gitignored) with all configs
ranked by mean R^2, plus per-fold dominance vs ridge for the top-ranked
candidate."
```

---

## Sprint completion checklist

After Task 3 (Phase 1 success path) **OR** Task 5 (Phase 2 path), exactly one of these is true:

- [ ] **Phase 1 success:** GCN with signed edge weights beat ridge on ≥ 3 of 5 folds → tasks 4–5 skipped.
- [ ] **Phase 2 success:** the best swept config beat ridge on ≥ 3 of 5 folds → reported in `results/metrics_ptt_average_sweep.json`.
- [ ] **Honest negative:** neither phase cleared the bar; the sweep table is preserved in the JSON for the paper.

In all three, `results/metrics_ptt_average*.json` is up to date and reproducible from `SEED=42`.
