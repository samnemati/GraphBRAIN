# Design — First End-to-End GraphMind Run (MoCA + PTT)

**Date:** 2026-04-30
**Project:** GraphMind (ABC Healthy Aging GNN regression)
**Author:** Samaneh Nemati + Claude
**Status:** Approved (awaiting implementation plan)

---

## Goal

Take the existing GraphMind scaffold from "untested stub code" to "first real cross-validated regression results on all 305 subjects" for both target variables (`MoCA_T`, `PTT_Average`), with a ridge regression baseline for comparison. After this sprint, Samaneh can look at concrete numbers (MAE / R² / Pearson r per fold, predicted-vs-actual scatter plots) and decide whether the GNN is learning anything useful, or whether the 2025 npj AI paper's "linear models are competitive" finding applies here too.

We are deliberately **not** trying to produce final, paper-ready numbers — this sprint produces a first datapoint with a known-honest pipeline, which we can then improve.

## Background

The repository scaffold described in `CLAUDE.md` already exists with ~600 lines of stub Python across `data/`, `models/`, `training/`, `baselines/`, and `interpretation/`. None of it has been run. Initial code review surfaced three concrete bugs and one structural concern; we expect more to surface once we exercise the pipeline against a real `.mat` file.

The dataset (Session 1) lives at `/Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/Brain_Data/ABC_MATfiles_session1/` with 306 subjects (305 of whom also have behavioral records). The behavioral file is `/Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/Behavioral_Data/ABC_PTT_WIN_MoCA.xlsx`.

## Non-Goals

This sprint deliberately does **not** include:

- Normalization beyond per-feature z-score on node inputs — no per-subject scaling, no connectivity-edge normalization beyond what ridge already does
- Hyperparameter tuning, early stopping, learning-rate scheduling — first pass uses `config.py` defaults
- GAT model run (the GAT is for interpretability; we'll add it once we know the GCN baseline numbers)
- GNNExplainer / attention-weight extraction (a separate "interpretability" sprint)
- Session 2 longitudinal data
- WIN_Threshold_Average and Age targets (the latter is a sanity check, deferred)
- A `pytest` test suite (YAGNI; the smoke-test script is sufficient until we have multiple modules to regression-test)
- Filling in `notebooks/exploration.ipynb` (the inspection script + scatter plots are the equivalent for now)
- GPU training (will revisit only if CPU is unworkably slow)
- Any tweaks to thresholding (`THRESHOLD_PERCENT`) or atlas — first pass uses what's in `config.py` exactly
- Adding node features beyond FA and MD — graph theory measures (node degree, clustering coefficient), VBM, PALF, etc. are all deferred to a later sprint

## Scope (11 steps, broken into 3 phases)

### Phase A — Verify and unblock (~30 min)

#### 1. Set node features to FA + MD

Edit `config.py`: change `NODE_FEATURES = ["fa", "md", "vbm_gm", "vbm_wm", "palf"]` to `NODE_FEATURES = ["fa", "md"]`. Rationale: white-matter microstructure features paired with a white-matter atlas; functional information is captured in edges. Also update the corresponding line in `CLAUDE.md` ("Stack `[fa, md, vbm_gm, vbm_wm, palf]` → shape [189 × 5]" → "Stack `[fa, md]` → shape [189 × 2]") and the "Node features" row of the design-decisions table.

#### 2. Inspect one `.mat` file

Add `scripts/inspect_mat.py`. It loads the first available subject (alphabetically) from `MAT_DIR_S1` using `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` and prints, both to stdout and to `scripts/mat_structure.txt`:

- Top-level keys of the loaded dict (filtering out `__header__`, `__version__`, `__globals__`)
- For each of `rest_jhu`, `fa_jhu`, `md_jhu`, `vbm_gm_jhu`, `vbm_wm_jhu`, `palf_jhu` (we inspect all six even though we only use two — useful reference for future sprints):
  - Python type
  - If `numpy.ndarray`: shape and `dtype`
  - If a `mat_struct`: the list of field names (`._fieldnames`) plus type/shape of each field

The `scripts/` directory itself is committed (so `smoke_test.py` in step 5 stays around as a runnable dev tool), but `scripts/mat_structure.txt` is gitignored — it's a generated artifact we can regenerate at any time.

#### 3. Reconcile reality vs. CLAUDE.md (only for FA and MD)

Based on step 2's output, decide for `fa_jhu` and `md_jhu` whether each is:

- **Form A:** a plain `numpy.ndarray` of shape `(189,)` or `(1, 189)` — access as `mat[key]`
- **Form B:** a `mat_struct` with a `.mean` field of shape `(189,)` or `(1, 189)` — access as `mat[key].mean`

Update **both** `data/loader.py` and (if needed) the relevant lines of `CLAUDE.md` so docs match code. We do not need to fix loader access for VBM/PALF since `NODE_FEATURES` no longer references them, but we record their forms in `mat_structure.txt` for the future.

#### 4. Fix three known bugs

| File | Issue | Fix |
|---|---|---|
| `data/graph_builder.py` | Uses `pd.isna(y_val)` at line 58 but never imports `pandas`. Will raise `NameError`. | Add `import pandas as pd`. |
| `data/loader.py` | Line 50 uses `.mean` access — verify against step 3. | Apply the form chosen in step 3 for FA and MD. |
| `data/dataset.py` | The `BrainGraphDataset(InMemoryDataset)` class instantiates with `root=None` and skips PyG's `process()` lifecycle. Fragile and unnecessary. | Replace with a plain function `build_dataset(target_col: str, mat_dir=MAT_DIR_S1, use_cache=True) -> list[Data]`. PyG's `DataLoader` accepts a plain `list[Data]`. Update `main.py` to use it. |

#### 5. One-subject smoke test

Add `scripts/smoke_test.py` that loads one subject, builds one PyG `Data`, instantiates `GCNRegression`, runs one forward pass, and asserts:

- `data.x.shape == (189, 2)` (FA + MD)
- `data.x.dtype == torch.float32`
- `data.edge_index.shape[0] == 2`
- `data.edge_attr.shape == (data.edge_index.shape[1], 1)`
- `data.y.shape == (1,)`
- Output of `GCNRegression(num_node_features=2)(...)` is a 1-D tensor of length 1

Prints `SMOKE TEST PASSED` only on success. Acts as the gate before Phase B.

### Phase B — Scale up and reproducibility (~45 min coding + ~5 min first data load)

#### 6. Load all 305 subjects with disk caching

Inside `build_dataset(target_col, use_cache=True)`:

- Compute a cache key from `target_col` plus a hash of `tuple(NODE_FEATURES)`, `THRESHOLD_PERCENT`, `ATLAS`, `EDGE_TYPE` — anything in `config.py` that affects graph contents.
- Cache file path: `results/cache/dataset_<target>_<hash>.pkl`
- If cache exists and `use_cache=True`, load via `pickle` and return.
- Otherwise: iterate `MAT_DIR_S1`, call `load_subject_mat` + `build_graph` per subject, drop subjects with missing target, save the resulting `list[Data]` to the cache file, return.
- Print a one-line summary: `Loaded N subjects for target '<target>' (X cached, Y from .mat)`.

`results/cache/` is added to `.gitignore`.

#### 7. Add reproducibility seeds

Add a single `set_seed(seed: int)` helper in `training/train.py` (or a new `training/utils.py`) that calls:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)` (no-op on CPU)
- `torch.backends.cudnn.deterministic = True` (only if `torch.cuda.is_available()`)

Call `set_seed(SEED)` at the top of `cross_validate(...)` and at the top of `main()` in `main.py`. Now reruns of the same target with the same config produce the same numbers.

### Phase C — Train, baseline, evaluate (~30 min coding + 30–120 min runtime)

#### 8. Wire up the ridge baseline

Confirm `baselines/ridge_regression.py` runs end-to-end. The current `main.py` already has a `--baseline_only` branch; verify it works after the `build_dataset` refactor. Print per-fold metrics and mean ± std. The baseline operates on flattened upper-triangle of `rest_jhu['r']` (already implemented) — no change needed to that logic.

#### 9. Run GCN cross-validation on `MoCA_T` (with per-fold node-feature standardization)

`python main.py --target MoCA_T --model gcn`. Uses `config.py` defaults: 5-fold, 200 epochs/fold, hidden_dim=64, 2 layers, dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=16, **num_node_features=2 (FA + MD)**. Prints per-fold and mean ± std for MAE, R², Pearson r.

Modify `training/train.cross_validate(...)` to:

1. **Standardize node features per fold (no leakage).** Inside the fold loop, *before* model training:
   - Stack `x` from all *training* `Data` objects → shape `(N_train × 189, 2)`
   - Compute per-column mean and std (so `mean.shape == std.shape == (2,)` — one per feature)
   - Apply `(x - mean) / std` to **both** training and validation `Data.x` tensors (using a copy, so the underlying cached dataset is untouched and reusable across folds)
   - The fitted statistics come from the training fold only — validation fold uses the same train-fitted `mean` / `std`. This is the standard leakage-free protocol.
2. **Return concatenated validation predictions + targets across all folds** (in fold order), so step 10 can plot them.

Implementation note: the dataset cache from step 6 stays raw (un-standardized). Standardization is a fold-level operation, not a cache-level operation, which means the cache is reusable if we later try a different normalization scheme.

#### 10. Save predicted-vs-actual scatter plot

Use existing `training/evaluate.plot_predictions(...)` to save `results/scatter_moca_gcn.png` and `results/scatter_moca_ridge.png` (concatenated validation predictions across folds, with overall MAE / R² / r in the title). Also dump the raw fold-level metrics to `results/metrics_moca.json` for later comparison.

#### 11. Repeat 8–10 for `PTT_Average`

Same code path, just `--target PTT_Average`. The build_dataset call drops the 28 subjects with missing PTT, leaving ~277 subjects. Outputs go to `results/scatter_ptt_gcn.png`, `results/scatter_ptt_ridge.png`, `results/metrics_ptt.json`.

If GCN training on MoCA in step 9 takes longer than ~30 min, we pause and reconsider before step 11 (e.g. drop epochs from 200 to 100, add early stopping, or accept that PTT will take a similar amount of time).

## Architecture / Components Touched

```
GraphMind/
├── scripts/                              [NEW — directory committed]
│   ├── inspect_mat.py                    [NEW — committed]
│   ├── mat_structure.txt                 [NEW — generated, gitignored]
│   └── smoke_test.py                     [NEW — committed; dev tool]
├── data/
│   ├── loader.py                         [MODIFIED — fix .mean access if needed]
│   ├── graph_builder.py                  [MODIFIED — add pandas import]
│   └── dataset.py                        [REWRITTEN — class → build_dataset() function with caching]
├── training/
│   ├── train.py                          [MODIFIED — set_seed(), per-fold node-feature standardization, return preds+targets from cross_validate]
│   └── evaluate.py                       [unchanged unless plot_predictions needs minor tweaks]
├── baselines/
│   └── ridge_regression.py               [unchanged in logic; verify it runs]
├── config.py                             [MODIFIED — NODE_FEATURES = ["fa", "md"]]
├── main.py                               [MODIFIED — use build_dataset(), call set_seed(), drive ridge + GCN, save plots + metrics JSON]
├── results/
│   ├── cache/                            [NEW — gitignored, generated]
│   ├── scatter_moca_gcn.png              [NEW — generated]
│   ├── scatter_moca_ridge.png            [NEW — generated]
│   ├── metrics_moca.json                 [NEW — generated]
│   ├── scatter_ptt_gcn.png               [NEW — generated]
│   ├── scatter_ptt_ridge.png             [NEW — generated]
│   └── metrics_ptt.json                  [NEW — generated]
├── .gitignore                            [MODIFIED — add scripts/mat_structure.txt, results/cache/, __pycache__/, *.pyc, .ipynb_checkpoints/]
└── CLAUDE.md                             [MODIFIED — node-features description (step 1) and any .mat-structure mismatches found in step 2]
```

No changes to: `models/gcn_regression.py`, `models/gat_regression.py`, `interpretation/*`, `requirements.txt`, `README.md`.

## Success Criteria

After this sprint, **all** of the following are true:

1. `python scripts/smoke_test.py` runs to completion and prints `SMOKE TEST PASSED`.
2. `python main.py --target MoCA_T --baseline_only` produces ridge metrics on MoCA across 5 folds.
3. `python main.py --target MoCA_T --model gcn` produces GCN metrics on MoCA across 5 folds, plus `results/scatter_moca_gcn.png` and `results/metrics_moca.json`.
4. Same as 2 + 3 for `PTT_Average`.
5. Re-running steps 2–4 with the same config produces **bit-identical** numbers (proof seeds work).
6. The dataset cache (`results/cache/dataset_<target>_<hash>.pkl`) exists and a second run of `main.py` for the same target loads in under 10 seconds.

We do **not** require the GCN to beat the ridge baseline — that's a research finding, not a sprint deliverable.

## Risks & Open Questions

- **Risk: CPU training time unknown.** 5-fold × 200 epochs × ~244 training graphs per fold could be 10 minutes or 2 hours depending on machine. **Mitigation:** if step 9 is still running at the 30-minute mark, pause and reduce epochs to 100 (one-line change in `config.py`). Keep `LEARNING_RATE` and other hyperparameters fixed for this first run.
- **Risk: GCN may not beat ridge.** R² could be near 0 or negative on MoCA — directly consistent with the 2025 npj AI paper. **Mitigation:** none needed; this is a real research finding to report honestly. The seed-determinism check in success criterion 5 ensures we are not chasing noise.
- **Risk: First-fold metrics on small validation sets (~61 subjects per fold) will be noisy.** A single fold's r could swing ±0.15 just from sampling. **Mitigation:** report mean ± std across folds, not best fold; use the concatenated-prediction scatter for the visual.
- **Risk: Some subjects may fail to load** (corrupted `.mat`, missing field). The current `loader.py` already wraps `load_subject_mat` in try/except and prints a warning. This is fine for sprint purposes; we'll log the count of failures.
- **Risk: scipy `loadmat` may not parse some `_jhu` fields the way we expect.** Mitigation: step 2 explicitly enumerates every field; we document anomalies before writing loader fixes.
- **Open question — subject loading:** does `fa_jhu` or `md_jhu` ever fail to load for some subject? If yes, the loader will skip them via the try/except. We'll know the survivor count after step 6.
- **Resolved — feature scale.** FA values are typically in `[0, 1]` (often 0.2–0.7 in WM) while MD values are in mm²/s (~5×10⁻⁴ to 1.5×10⁻³), a ~1000× scale gap. **Decision:** standardize node features per-fold using only training-fold statistics (option b above). Implemented inside `cross_validate(...)` per step 9. This is the standard leakage-free protocol in the ML connectomics literature. We did *not* choose per-subject z-scoring because for an aging-regression task it would erase biologically meaningful between-subject differences in mean FA / MD.

## Next Steps After This Sprint

In rough priority order:

1. **GCN tuning sprint** if first-pass GCN ≈ ridge or worse: try standardizing node features, increase/decrease layers, try GraphSAGE or GIN.
2. **GAT + interpretability sprint:** train GAT on the better target, extract attention weights, run GNNExplainer, identify top regions/connections.
3. **Multi-target / multi-task sprint:** add WIN_Threshold_Average, joint MoCA + PTT prediction.
4. **Paper draft:** introduction, methods (already 80% in this spec), results table.

We will brainstorm and design each sprint separately before touching code.
