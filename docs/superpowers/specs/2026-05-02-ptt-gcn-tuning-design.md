# Design — PTT GCN Tuning Sprint

**Date:** 2026-05-02
**Project:** GraphMind (ABC Healthy Aging GNN regression)
**Author:** Samaneh Nemati + Claude
**Status:** Approved (awaiting implementation plan)

---

## Goal

Get the GCN model on `PTT_Average` to match or beat the ridge-regression baseline. Ridge per-fold R² values from the prior sprint were `[0.424, 0.246, 0.329, 0.231, 0.102]` (mean 0.267).

**Decision criterion: per-fold dominance.** The GCN must beat ridge on at least **3 of the 5 folds** (compared positionally, fold 1 vs fold 1 etc.) — this is more robust than mean comparison alone, since a single lucky fold could lift the GCN's mean above ridge while the model is actually worse on most splits. This rule has the same spirit as a paired sign test.

A wrinkle: the GCN runs on 229 subjects (FA/MD must be finite) while ridge runs on 238 (connectivity has no NaN-filtering issue). The 5-fold splits therefore aren't perfectly paired. We accept this — the difference is 9 subjects out of ~46 per fold and doesn't materially shift the comparison; a strictly paired comparison would require re-running ridge on the GCN-eligible subset, which is overkill for a sprint-level decision.

Pearson r and MAE are reported but are not the gate — R² is the most sensitive of the three metrics to model bias and is the standard regression-quality measure.

Two phases. Phase 1 is a one-shot architectural fix that we expect to be the biggest single contribution. Phase 2 is a small targeted hyperparameter sweep that runs only if Phase 1 doesn't clear the bar.

## Background

In the just-completed first end-to-end sprint, the GCN underperformed ridge on PTT_Average:

| Method | MAE | R² | Pearson r |
|---|---|---|---|
| Ridge | 8.796 ± 0.863 | 0.267 ± 0.107 | 0.543 ± 0.076 |
| GCN | 8.619 ± 0.728 | 0.212 ± 0.187 | 0.525 ± 0.155 |

Ridge wins on R²; GCN narrowly wins on MAE; r is essentially tied. Inspecting the model, the cause is structural: `models/gcn_regression.py:42` calls `conv(x, edge_index)` without `edge_weight`. The connectivity strength values stored in `data.edge_attr` never reach message passing. The GCN sees only edge presence; ridge sees full edge weights through the flattened upper triangle. That's the immediate signal asymmetry to fix.

## Non-Goals

- Tuning for `MoCA_T` (the prior sprint showed both methods struggle; deferred until we have a better setup for PTT).
- Conv types beyond `GraphConv` (no GraphSAGE, GIN, EdgeConv this sprint).
- Hidden-dim sweep (we keep `HIDDEN_DIM = 64` from `config.py` for both phases).
- Batch-size sweep, weight-decay sweep, optimizer change.
- Statistical-significance tests across sweep configs (mean R² ranking is enough for first-pass model selection).
- Fold-5 outlier investigation from the prior sprint.
- Publishing a separate file for ridge sweep — ridge runs as the baseline only, no tuning.

## Scope

### Phase 1 — Signed edge weights (one-shot, ~5 min coding + ~12 min runtime)

**Architectural problem.** `GCNConv` uses symmetric normalization (`D^(-1/2) · A · D^(-1/2)`) that mathematically requires non-negative edge weights. The Pearson correlations stored in `data.edge_attr` are signed (typically −0.3 to +0.9 in the surviving top-15% edges). Taking `abs(edge_attr)` would erase the sign distinction between correlated and anti-correlated regions — biologically meaningful for hearing, since negative correlations between functional modules carry different information than positive ones. We therefore replace `GCNConv` with a conv that handles signed weights natively.

**The change.** Replace `GCNConv` with `torch_geometric.nn.GraphConv` in `models/gcn_regression.py`. `GraphConv`'s formula is

> x_i' = W₁ · x_i + W₂ · Σⱼ (e_ij · x_j)

— a weighted sum of neighbor features with a separate self-transform, no symmetric normalization. Signed weights are mathematically valid: positive edges add the neighbor's features, negative edges subtract them, scaled by magnitude. Pass `edge_weight=data.edge_attr.squeeze(-1)` (raw signed Pearson r) into both conv calls in `forward()`.

**File-naming decision.** Keep `models/gcn_regression.py` and class `GCNRegression` as the names — they reasonably cover any "graph conv for regression." Update the module docstring to call out that the conv layer has been changed and why.

**Run.** `python main.py --target PTT_Average --model gcn` (the dataset cache stays valid since we haven't touched data layout). Compare GCN per-fold R² (5 values) against ridge per-fold R² `[0.424, 0.246, 0.329, 0.231, 0.102]` positionally.

- If GCN R² > ridge R² on **≥ 3 of 5 folds** → sprint done. Report results, move to GAT/interpretability sprint.
- Otherwise → Phase 2.

### Phase 2 — Hyperparameter sweep (only if Phase 1 fails, ~75 min runtime)

**Add early stopping** to `training/train.cross_validate(...)`:
- Patience: 20 epochs without improvement on validation MAE.
- "Best" is tracked by lowest validation MAE seen so far.
- On stop or end-of-epochs, restore the best-MAE model weights before the final evaluation pass.
- Existing 200-epoch ceiling stays as a hard cap.

**Sweep 12 configs** = lr × dropout × layers:
- `lr ∈ {1e-4, 5e-4, 1e-3}`
- `dropout ∈ {0.2, 0.5}`
- `num_layers ∈ {2, 3}`
- Other hyperparameters fixed at `config.py` defaults (`hidden_dim=64`, `weight_decay=1e-4`, `batch_size=16`, `seed=42`).
- Edge-weight handling stays on (from Phase 1).

**Driver.** A new `scripts/sweep_ptt_gcn.py` runs all 12 configs sequentially, reports per-config mean ± std for MAE / R² / r across the 5 folds, and writes a single `results/metrics_ptt_average_sweep.json` containing every config + per-fold details. The script does *not* modify `config.py`; it overrides hyperparameters in-process by:

1. Passing `dropout` and `num_layers` directly to `GCNRegression(...)` via `model_kwargs`.
2. Adding an optional `lr` parameter to `cross_validate(...)` that defaults to `config.LEARNING_RATE` and is plumbed into the `Adam` optimizer construction. The default behaviour from the prior sprint is preserved when `lr` is omitted.

This is a small, contained change to `cross_validate`'s signature on top of the early-stopping addition.

**Decision.** Pick the config with the highest mean R² as the best candidate. Then apply the per-fold dominance rule: if that config's per-fold R² beats ridge on ≥ 3 of 5 folds → success. Otherwise → honest negative result; write up as-is and move to GAT sprint.

(Mean R² is used to *rank* the 12 configs because per-fold dominance produces ties, and we need a single ordering. The per-fold rule is then applied only to the top-ranked candidate as the final gate.)

## Architecture / Components Touched

```
GraphMind/
├── models/
│   └── gcn_regression.py                 [Phase 1 — GCNConv → GraphConv, pass edge_weight]
├── training/
│   └── train.py                          [Phase 2 — early stopping + best-weights restore]
├── scripts/
│   └── sweep_ptt_gcn.py                  [Phase 2 — NEW; runs 12-config sweep]
└── results/
    ├── scatter_ptt_average_gcn.png       [updated by Phase 1 re-run]
    ├── metrics_ptt_average.json          [updated by Phase 1 re-run]
    └── metrics_ptt_average_sweep.json    [Phase 2 only — all 12 configs]
```

No changes to: `data/*`, `config.py`, `main.py`, `baselines/*`, `interpretation/*`. (`main.py` continues to work because the new `lr` parameter on `cross_validate` is optional — main.py keeps calling it without `lr` and inherits `config.LEARNING_RATE`.)

## Success Criteria

After this sprint, **exactly one** of the following is true:

1. **Phase 1 success:** with edge weights, GCN beats ridge per-fold R² on ≥ 3 of 5 folds; sprint stops at Phase 1.
2. **Phase 2 success:** the best-mean-R² config among the 12 sweep configs beats ridge per-fold R² on ≥ 3 of 5 folds.
3. **Honest negative:** neither phase reaches the per-fold-dominance bar; we have the sweep table for the paper, showing we tried and the GCN architecture isn't the bottleneck for this signal.

In all three cases, `results/metrics_ptt_average*.json` and the scatter PNGs are up-to-date and reproducible from `SEED=42`.

## Risks & Open Questions

- **Risk: signed messages may destabilize training.** Negative edges produce subtractive messages; combined with ReLU activations and normalization layers, this can in theory cause representation collapse. Mitigation: BatchNorm layers in the existing model class normalize activations; if Phase 1 produces NaNs, that's a code bug not a math one (we'd add gradient clipping and re-run).
- **Risk: GraphConv with no normalization may amplify activations.** Without `D^(-1/2)` scaling, nodes with many high-magnitude edges get larger messages than nodes with few edges. The BatchNorm after each conv handles this in practice. We do not add a custom normalization step in this sprint.
- **Risk: Phase 2 outlier folds.** The prior GCN run had a fold-5 outlier (R² = -0.16) that dragged the mean. Early stopping should mitigate this somewhat (the model that the outlier fold trains gets stopped before it overfits noise), but if fold 5 stays bad even in the best swept config, that's a data signal — leave the investigation for a later sprint.
- **Open question:** does `GraphConv` use mean or sum aggregation by default? Default is `aggr='add'`, which is the right semantic ("sum of weighted neighbor features") for our signed-weight case. We'll leave it at the default.
- **Open question — runtime:** 12 configs × 5 folds with early stopping should average ~80 epochs/fold (vs the 200-epoch hard cap). Estimated total ~75 min on CPU; max 3 hours if early stopping rarely triggers. Still single-session-friendly.

## Next Steps After This Sprint

If Phase 1 or Phase 2 succeeds: move to the **GAT + interpretability sprint** as planned — train a GAT variant on PTT_Average, extract attention weights and run GNNExplainer to identify the regions and connections driving prediction.

If both phases fail: still move to the GAT sprint, but with the negative GCN result documented as a real comparison point. The interpretability story doesn't require GCN ≥ ridge.
