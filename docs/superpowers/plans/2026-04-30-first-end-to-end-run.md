# First End-to-End GraphMind Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the existing GraphMind scaffold from "untested stub code" to "first cross-validated regression results on all ~305 subjects" for both `MoCA_T` and `PTT_Average`, with a ridge baseline for comparison.

**Architecture:** Three phases — (A) verify and unblock the scaffold by inspecting one `.mat` file and fixing known bugs; (B) scale up to all subjects with disk caching and reproducibility seeds; (C) train GCN with per-fold node-feature standardization, run ridge baseline, and save results (per-fold metrics JSON + concatenated-prediction scatter plots) for both targets.

**Tech Stack:** Python 3, scipy.io for `.mat` loading, PyTorch, PyTorch Geometric (GCNConv + global_mean_pool), scikit-learn (RidgeCV, KFold, StandardScaler), pandas, matplotlib.

**Spec:** [docs/superpowers/specs/2026-04-30-first-end-to-end-run-design.md](../specs/2026-04-30-first-end-to-end-run-design.md)

---

## File Structure

```
GraphMind/
├── scripts/                              [NEW dir, committed]
│   ├── inspect_mat.py                    [Task 3 — one-shot .mat structure inspector]
│   ├── mat_structure.txt                 [Task 3 — generated, gitignored]
│   └── smoke_test.py                     [Task 8 — single-subject end-to-end smoke test]
├── data/
│   ├── loader.py                         [Task 5 — fix FA/MD access form]
│   ├── graph_builder.py                  [Task 4 — add pandas import]
│   └── dataset.py                        [Task 6 — class → build_dataset() function]
├── training/
│   └── train.py                          [Task 10 — set_seed; Task 11 — per-fold standardization + return preds]
├── baselines/
│   └── ridge_regression.py               [Task 12 — return preds/targets in addition to metrics]
├── config.py                             [Task 1 — NODE_FEATURES = ["fa", "md"]]
├── main.py                               [Task 7 — use build_dataset(); Task 12 — driver for ridge + GCN + plots + JSON]
├── results/
│   ├── cache/                            [Task 9 — generated, gitignored]
│   ├── scatter_moca_gcn.png              [Task 14]
│   ├── scatter_moca_ridge.png            [Task 14]
│   ├── metrics_moca.json                 [Task 14]
│   ├── scatter_ptt_gcn.png               [Task 15]
│   ├── scatter_ptt_ridge.png             [Task 15]
│   └── metrics_ptt.json                  [Task 15]
├── .gitignore                            [Task 2 — Python entries + generated artifacts]
└── CLAUDE.md                             [Task 1 — node features description; Task 5 — .mat field forms if mismatch]
```

Each task ends with one logical commit. A few tasks (3, 5) include investigation steps before the commit because the implementation depends on what the inspection reveals.

---

## Task 1: Set node features to FA + MD; update CLAUDE.md

**Files:**
- Modify: `config.py:22`
- Modify: `CLAUDE.md` (two locations)

- [ ] **Step 1: Edit `config.py:22`**

Change:
```python
NODE_FEATURES = ["fa", "md", "vbm_gm", "vbm_wm", "palf"]
```
to:
```python
NODE_FEATURES = ["fa", "md"]
```

- [ ] **Step 2: Edit `CLAUDE.md` — design-decisions table row "Node features"**

Find the row that begins:
```
| Node features | `fa_jhu`, `md_jhu`, `vbm_gm_jhu`, `vbm_wm_jhu`, `palf_jhu` + graph theory measures |
```

Replace the cell content with:
```
| Node features | `fa_jhu`, `md_jhu` | DTI white-matter microstructure measures paired with the white-matter atlas; functional info is captured in edges |
```

- [ ] **Step 3: Edit `CLAUDE.md` — Graph Construction "Node feature matrix X" line**

Find:
```
- **Node feature matrix X:** Stack `[fa, md, vbm_gm, vbm_wm, palf]` → shape [189 × 5], optionally add graph theory measures (node degree, clustering coefficient)
```

Replace with:
```
- **Node feature matrix X:** Stack `[fa, md]` → shape [189 × 2]
```

- [ ] **Step 4: Verify**

Run: `grep -n "NODE_FEATURES" config.py`
Expected: shows `NODE_FEATURES = ["fa", "md"]`

Run: `grep -n "vbm_gm.*vbm_wm.*palf" CLAUDE.md || echo "OK - none found"`
Expected: prints `OK - none found` (the only references that should remain are deferred-feature mentions, which use commas not direct bracket notation)

- [ ] **Step 5: Commit**

```bash
git add config.py CLAUDE.md
git commit -m "config: narrow NODE_FEATURES to FA + MD only

White-matter microstructure features paired with white-matter atlas;
functional information is preserved in edges (rest_jhu['r']).
Drops VBM and PALF from node features for cleaner first-pass model."
```

---

## Task 2: Update `.gitignore` for Python entries + generated artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append Python and project-specific entries to `.gitignore`**

Append the following block to the end of `.gitignore`:

```
# === Python ===
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
*.egg-info/

# === GraphMind generated artifacts ===
scripts/mat_structure.txt
results/cache/
results/*.png
results/*.json
!results/.gitkeep
```

(The `!results/.gitkeep` exception keeps the empty `results/` directory tracked while ignoring all generated content inside it.)

- [ ] **Step 2: Verify `results/.gitkeep` exists; create if not**

Run: `ls /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/.gitkeep 2>&1 || touch /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/.gitkeep`
Expected: file exists or is created.

- [ ] **Step 3: Verify `git status` is clean for these patterns**

Run: `git status --short results/ scripts/`
Expected: only `results/.gitkeep` shows up if newly created; no pyc/__pycache__/cache files leak in.

- [ ] **Step 4: Commit**

```bash
git add .gitignore results/.gitkeep
git commit -m "gitignore: add Python and GraphMind generated-artifact patterns"
```

---

## Task 3: Inspect one `.mat` file and record its structure

**Files:**
- Create: `scripts/inspect_mat.py`
- Create (generated): `scripts/mat_structure.txt`

- [ ] **Step 1: Create `scripts/inspect_mat.py`**

```python
"""
inspect_mat.py - One-shot inspector for the structure of a single subject .mat file.

Loads the first available .mat from MAT_DIR_S1, prints (and writes to mat_structure.txt)
the type, shape, and (if a struct) field names of every key relevant to the GraphMind
pipeline. Run once per significant change in the source data.
"""

import os
import sys
import numpy as np
import scipy.io as sio

# Add repo root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAT_DIR_S1

KEYS_OF_INTEREST = [
    "rest_jhu",
    "fa_jhu", "md_jhu",
    "vbm_gm_jhu", "vbm_wm_jhu", "palf_jhu",
]


def describe(value, prefix=""):
    """Return a string describing `value` (its type, shape, fields)."""
    lines = []
    t = type(value).__name__
    if isinstance(value, np.ndarray):
        lines.append(f"{prefix}type=ndarray  shape={value.shape}  dtype={value.dtype}")
    elif hasattr(value, "_fieldnames"):
        lines.append(f"{prefix}type=mat_struct  fields={list(value._fieldnames)}")
        for fname in value._fieldnames:
            sub = getattr(value, fname)
            lines.append(describe(sub, prefix=prefix + f"  .{fname}: "))
    else:
        lines.append(f"{prefix}type={t}  value={value!r}")
    return "\n".join(lines)


def main():
    mat_files = sorted(f for f in os.listdir(MAT_DIR_S1) if f.endswith(".mat"))
    if not mat_files:
        raise SystemExit(f"No .mat files in {MAT_DIR_S1}")
    target = os.path.join(MAT_DIR_S1, mat_files[0])
    print(f"Inspecting: {target}")
    mat = sio.loadmat(target, squeeze_me=True, struct_as_record=False)

    out_lines = [f"# .mat structure inspection — {mat_files[0]}", ""]
    out_lines.append("## Top-level keys")
    keys = sorted(k for k in mat.keys() if not k.startswith("__"))
    for k in keys:
        out_lines.append(f"  {k}")
    out_lines.append("")

    out_lines.append("## Keys of interest")
    for k in KEYS_OF_INTEREST:
        out_lines.append(f"### {k}")
        if k not in mat:
            out_lines.append("  MISSING")
        else:
            out_lines.append(describe(mat[k], prefix="  "))
        out_lines.append("")

    text = "\n".join(out_lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "mat_structure.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the inspection script**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python scripts/inspect_mat.py`
Expected: prints the structure of one subject's `.mat` file and writes `scripts/mat_structure.txt`. Confirms `rest_jhu` has fields `r` and `label` (per CLAUDE.md). Records the form of each `_jhu` key.

- [ ] **Step 3: Read the output and decide FA/MD access form**

Open `scripts/mat_structure.txt`. For each of `fa_jhu` and `md_jhu`, note whether it is:
- **Form A:** `type=ndarray  shape=(189,)` or `(1, 189)` — access in loader as `mat["fa_jhu"]`
- **Form B:** `type=mat_struct  fields=['mean', ...]` with a `.mean` of shape `(189,)` — access as `mat["fa_jhu"].mean`

Record the chosen form (A or B) — Task 5 depends on this.

- [ ] **Step 4: Commit (script only — `mat_structure.txt` is gitignored)**

```bash
git add scripts/inspect_mat.py
git commit -m "scripts: add one-shot .mat structure inspector

Records the type/shape/fields of rest_jhu and the per-region feature
keys for one subject. Output to scripts/mat_structure.txt (gitignored)."
```

---

## Task 4: Fix `data/graph_builder.py` — add missing `pandas` import

**Files:**
- Modify: `data/graph_builder.py:11-14`

- [ ] **Step 1: Add `import pandas as pd` to the imports block**

Edit the file. Current top of file:
```python
import numpy as np
import torch
from torch_geometric.data import Data
from config import THRESHOLD_PERCENT, N_REGIONS
```

Change to:
```python
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from config import THRESHOLD_PERCENT, N_REGIONS
```

- [ ] **Step 2: Verify the import compiles**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "import data.graph_builder"`
Expected: no output (success). If `ModuleNotFoundError` for any package, install it.

- [ ] **Step 3: Commit (combined with Task 5)**

Do not commit yet — Task 5 fixes another file in the same logical change ("data layer bug fixes"). Combined commit at the end of Task 5.

---

## Task 5: Fix `data/loader.py` — correct FA/MD access form

**Files:**
- Modify: `data/loader.py:48-53`
- Modify: `CLAUDE.md` — only if step 3 of Task 3 revealed a documentation mismatch

- [ ] **Step 1: If Task 3 chose Form A (plain ndarray), edit `data/loader.py:46-53`**

Find:
```python
    feature_list = []
    for feat_name in NODE_FEATURES:
        key = f"{feat_name}_{ATLAS}"
        feat_data = mat[key].mean              # shape: (189,)
        feature_list.append(feat_data.flatten())
```

Replace with:
```python
    feature_list = []
    for feat_name in NODE_FEATURES:
        key = f"{feat_name}_{ATLAS}"
        feat_data = np.asarray(mat[key])       # shape: (189,) or (1, 189)
        feature_list.append(feat_data.flatten())
```

- [ ] **Step 2: If Task 3 chose Form B (mat_struct with .mean), keep the existing code**

The existing line `feat_data = mat[key].mean` is already correct. Add a `np.asarray(...)` wrapper for safety:

Change:
```python
        feat_data = mat[key].mean              # shape: (189,)
```

to:
```python
        feat_data = np.asarray(mat[key].mean)  # shape: (189,) or (1, 189)
```

(Apply only one of step 1 or step 2 — whichever matches Task 3's finding.)

- [ ] **Step 3: If Form A was chosen, update CLAUDE.md to reflect that fa_jhu/md_jhu are plain arrays**

Open `CLAUDE.md` and find the lines that describe `fa_jhu` / `md_jhu` as `1×189` (around the "Per-node features" section). If they're already documented as plain `1×189` arrays, the docs match Form A — no edit needed. If the docs say struct-with-`.mean`, update them to plain arrays. Only edit if the docs disagree with reality.

- [ ] **Step 4: Verify the loader works on one subject**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "
from data.loader import load_subject_mat
import os
from config import MAT_DIR_S1
fpath = os.path.join(MAT_DIR_S1, sorted(os.listdir(MAT_DIR_S1))[0])
conn, feats, labels = load_subject_mat(fpath)
print(f'connectivity={conn.shape}  features={feats.shape}  labels={len(labels)}')
"
```
Expected output (shape may vary by `(189,)` vs `(1, 189)`):
```
connectivity=(189, 189)  features=(189, 2)  labels=189
```
If `features` shape is `(189, 2)` exactly, success.

- [ ] **Step 5: Commit (combined with Task 4 — data layer bug fixes)**

```bash
git add data/loader.py data/graph_builder.py
# Only add CLAUDE.md if step 3 of this task edited it:
git add CLAUDE.md 2>/dev/null || true
git commit -m "data: fix loader feature access and graph_builder pandas import

- loader.py: load FA/MD per-region values using <chosen form> access
- graph_builder.py: add missing 'import pandas as pd' (used by pd.isna)
- CLAUDE.md: reconcile per-node-feature description with actual .mat layout (if needed)"
```

(Replace `<chosen form>` in the message with `plain ndarray` or `.mean field of mat_struct` per Task 3's finding.)

---

## Task 6: Replace `data/dataset.py` — class → `build_dataset()` function (no caching yet)

**Files:**
- Rewrite: `data/dataset.py`

- [ ] **Step 1: Rewrite `data/dataset.py` end-to-end**

Replace the entire file content with:

```python
"""
dataset.py - Build a list of PyG Data objects, one per subject, for a regression target.

This module deliberately does NOT subclass torch_geometric.data.InMemoryDataset:
- We don't need PyG's processed-files lifecycle (we have ~305 subjects, not millions).
- A plain list[Data] is what torch_geometric.loader.DataLoader actually wants.
- The function form is simpler to reason about and easier to test.

Disk caching is added in a later step (Task 9).
"""

from typing import List
from torch_geometric.data import Data
from data.loader import load_all_subjects, load_behavioral_data
from data.graph_builder import build_graph
from config import MAT_DIR_S1


def build_dataset(target_col: str, mat_dir: str = MAT_DIR_S1) -> List[Data]:
    """
    Load all subjects with both brain and behavioral data, build PyG Data
    objects for the given regression target, and return them as a list.

    Subjects with NaN target are dropped (build_graph returns None).
    Subjects whose .mat fails to load are skipped with a warning by the loader.
    """
    behavioral_df = load_behavioral_data()
    subject_data = load_all_subjects(mat_dir=mat_dir, behavioral_df=behavioral_df)

    graphs: List[Data] = []
    for s in subject_data:
        g = build_graph(s, target_col=target_col)
        if g is not None:
            graphs.append(g)

    print(f"Built {len(graphs)} graphs for target '{target_col}' "
          f"(from {len(subject_data)} subjects with .mat + behavioral data).")
    return graphs
```

- [ ] **Step 2: Verify the file compiles**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "from data.dataset import build_dataset; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 3: Commit (combined with Task 7 — main.py update)**

Do not commit yet — Task 7 updates `main.py` to consume the new function. Single commit at end of Task 7.

---

## Task 7: Update `main.py` to use `build_dataset()` instead of `BrainGraphDataset`

**Files:**
- Modify: `main.py:12, 45-52`

- [ ] **Step 1: Replace the `BrainGraphDataset` import**

Edit `main.py:12`. Change:
```python
from data.dataset import BrainGraphDataset
```
to:
```python
from data.dataset import build_dataset
```

- [ ] **Step 2: Replace the `dataset = BrainGraphDataset(...)` call**

Edit `main.py` around lines 45-52. Find:
```python
    # Load dataset
    dataset = BrainGraphDataset(target_col=args.target)
    num_features = len(NODE_FEATURES)

    model_kwargs = {"num_node_features": num_features}
    model_class = GCNRegression if args.model == "gcn" else GATRegression

    print(f"\n── {args.model.upper()} Cross-Validation ──")
    fold_metrics = cross_validate(model_class, dataset, model_kwargs, device)
```

Replace with:
```python
    # Load dataset
    dataset = build_dataset(target_col=args.target)
    num_features = len(NODE_FEATURES)

    model_kwargs = {"num_node_features": num_features}
    model_class = GCNRegression if args.model == "gcn" else GATRegression

    print(f"\n── {args.model.upper()} Cross-Validation ──")
    fold_metrics = cross_validate(model_class, dataset, model_kwargs, device)
```

(The only change is `BrainGraphDataset(target_col=args.target)` → `build_dataset(target_col=args.target)`. The rest of the function is unchanged for now; Task 12 expands it.)

- [ ] **Step 3: Verify `main.py` parses**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "import ast; ast.parse(open('main.py').read()); print('OK')"`
Expected: prints `OK`.

- [ ] **Step 4: Commit Tasks 6 + 7 together**

```bash
git add data/dataset.py main.py
git commit -m "data: replace BrainGraphDataset class with build_dataset() function

- Remove InMemoryDataset subclass (we don't need PyG's processed-files lifecycle)
- New module-level function build_dataset(target_col) -> list[Data]
- Update main.py to consume the function form

Disk caching is added in a follow-up commit."
```

---

## Task 8: Create `scripts/smoke_test.py` and run it (gate before Phase B)

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Create `scripts/smoke_test.py`**

```python
"""
smoke_test.py - End-to-end smoke test on a single subject.

Loads one subject .mat, builds one PyG Data object, instantiates GCNRegression,
runs one forward pass, and asserts shapes/dtypes are correct. Exits 0 on
success and prints SMOKE TEST PASSED. Run this before any Phase-B work.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAT_DIR_S1, NODE_FEATURES, N_REGIONS
from data.loader import load_subject_mat, load_behavioral_data
from data.graph_builder import build_graph
from models.gcn_regression import GCNRegression


def main():
    n_features = len(NODE_FEATURES)
    assert n_features == 2, f"Expected 2 node features (FA + MD), got {n_features}"

    behavioral_df = load_behavioral_data()
    mat_files = sorted(f for f in os.listdir(MAT_DIR_S1) if f.endswith(".mat"))

    # Find first subject that has behavioral data and a non-NaN MoCA_T
    subject_dict = None
    for fname in mat_files:
        sid = fname.replace(".mat", "")
        if sid not in behavioral_df.index:
            continue
        moca = behavioral_df.loc[sid, "MoCA_T"]
        if moca != moca:  # NaN check
            continue
        conn, feats, labels = load_subject_mat(os.path.join(MAT_DIR_S1, fname))
        subject_dict = {
            "subject_id": sid, "connectivity": conn, "node_features": feats,
            "roi_labels": labels, "behavioral": behavioral_df.loc[sid],
        }
        break
    assert subject_dict is not None, "No subject with brain + non-NaN MoCA_T behavioral data"

    print(f"Smoke test subject: {subject_dict['subject_id']}  "
          f"MoCA_T={subject_dict['behavioral']['MoCA_T']}")

    # Build graph
    data = build_graph(subject_dict, target_col="MoCA_T")
    assert data is not None

    # Shape / dtype assertions
    assert data.x.shape == (N_REGIONS, n_features), \
        f"x.shape {data.x.shape} != ({N_REGIONS}, {n_features})"
    assert data.x.dtype == torch.float32, f"x.dtype {data.x.dtype} != float32"
    assert data.edge_index.shape[0] == 2, f"edge_index.shape[0] {data.edge_index.shape[0]} != 2"
    assert data.edge_index.dtype == torch.long
    assert data.edge_attr.shape == (data.edge_index.shape[1], 1), \
        f"edge_attr.shape {data.edge_attr.shape} != ({data.edge_index.shape[1]}, 1)"
    assert data.y.shape == (1,), f"y.shape {data.y.shape} != (1,)"
    n_edges = data.edge_index.shape[1]

    # One forward pass
    model = GCNRegression(num_node_features=n_features)
    model.eval()
    batch = torch.zeros(N_REGIONS, dtype=torch.long)
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, batch)
    assert out.dim() == 1 and out.shape[0] == 1, f"output shape {out.shape} != (1,)"
    pred = float(out.item())
    actual = float(data.y.item())

    print(f"  edges={n_edges}  pred={pred:.3f}  actual={actual:.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python scripts/smoke_test.py`
Expected: prints subject info, edge count, untrained prediction, and `SMOKE TEST PASSED` as the last line. Exit code 0.

If it fails: investigate the assertion error before proceeding to Task 9. Common issues:
- Shape mismatch on `data.x`: check Task 5's loader fix
- `pd is not defined`: Task 4 wasn't completed
- `KeyError: 'MoCA_T'`: behavioral Excel column name has whitespace; strip in `load_behavioral_data`

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "scripts: add single-subject end-to-end smoke test

Loads one .mat, builds one PyG Data, runs one GCN forward pass,
and asserts shapes. Acts as the gate between Phase A (verify) and
Phase B (scale up)."
```

---

## Task 9: Add disk caching to `build_dataset()`

**Files:**
- Modify: `data/dataset.py`

- [ ] **Step 1: Replace `data/dataset.py` with caching-enabled version**

```python
"""
dataset.py - Build a list of PyG Data objects, one per subject, for a regression target.

Disk caching: pickled list[Data] in results/cache/dataset_<target>_<hash>.pkl.
Cache key = hash(target_col, NODE_FEATURES, THRESHOLD_PERCENT, ATLAS, EDGE_TYPE).
First run loads from .mat and writes the cache; subsequent runs read the pkl in
seconds. Standardization is NOT applied at cache time — it happens per-fold
inside cross_validate (so the cache stays reusable across normalization choices).
"""

import hashlib
import os
import pickle
from typing import List
from torch_geometric.data import Data
from data.loader import load_all_subjects, load_behavioral_data
from data.graph_builder import build_graph
from config import (MAT_DIR_S1, NODE_FEATURES, THRESHOLD_PERCENT,
                    ATLAS, EDGE_TYPE, RESULTS_DIR)


def _cache_key(target_col: str) -> str:
    """Stable hash over all config knobs that affect graph contents."""
    payload = repr((
        target_col,
        tuple(NODE_FEATURES),
        float(THRESHOLD_PERCENT),
        ATLAS,
        EDGE_TYPE,
    )).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _cache_path(target_col: str) -> str:
    cache_dir = os.path.join(RESULTS_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"dataset_{target_col}_{_cache_key(target_col)}.pkl")


def build_dataset(target_col: str, mat_dir: str = MAT_DIR_S1,
                  use_cache: bool = True) -> List[Data]:
    """
    Load all subjects with both brain and behavioral data, build PyG Data
    objects for the given regression target, and return them as a list.

    If `use_cache` and the cache file exists, load from pickle (fast).
    Otherwise iterate the .mat files, build graphs, save the cache, and return.
    """
    cache_path = _cache_path(target_col)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs for target '{target_col}' from cache: {cache_path}")
        return graphs

    behavioral_df = load_behavioral_data()
    subject_data = load_all_subjects(mat_dir=mat_dir, behavioral_df=behavioral_df)

    graphs: List[Data] = []
    for s in subject_data:
        g = build_graph(s, target_col=target_col)
        if g is not None:
            graphs.append(g)

    with open(cache_path, "wb") as f:
        pickle.dump(graphs, f)

    print(f"Built {len(graphs)} graphs for target '{target_col}' "
          f"(from {len(subject_data)} subjects). Cached to: {cache_path}")
    return graphs
```

- [ ] **Step 2: Verify cache round-trip with a quick script**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "
import time
from data.dataset import build_dataset
t0 = time.time(); g1 = build_dataset('MoCA_T'); t1 = time.time()
print(f'First load: {t1-t0:.1f}s, n_graphs={len(g1)}')
t0 = time.time(); g2 = build_dataset('MoCA_T'); t1 = time.time()
print(f'Cached load: {t1-t0:.1f}s, n_graphs={len(g2)}')
assert len(g1) == len(g2)
print('CACHE OK')
"
```
Expected: first load ~30–300 seconds (depending on disk), cached load < 10 seconds, both return the same graph count, prints `CACHE OK`.

- [ ] **Step 3: Commit**

```bash
git add data/dataset.py
git commit -m "data: add disk caching to build_dataset()

Cache key hashes target + NODE_FEATURES + THRESHOLD_PERCENT + ATLAS + EDGE_TYPE.
Cache stays raw (un-standardized) so it's reusable across normalization choices.
Cache files in results/cache/ (gitignored)."
```

---

## Task 10: Add `set_seed()` helper and call it from training entry points

**Files:**
- Modify: `training/train.py`
- Modify: `main.py`

- [ ] **Step 1: Add the `set_seed` helper at the top of `training/train.py`**

Edit `training/train.py`. Insert this function definition immediately after the existing imports block (around line 10, after the import lines but before `def train_one_epoch`):

```python
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

- [ ] **Step 2: Call `set_seed(SEED)` at the top of `cross_validate(...)`**

Find the line `def cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS):` and insert immediately inside the function body:

```python
def cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS):
    """Run k-fold cross-validation and return per-fold metrics."""
    set_seed(SEED)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    ...
```

- [ ] **Step 3: Call `set_seed(SEED)` at the top of `main()` in `main.py`**

Edit `main.py`. After the `args = parser.parse_args()` line, add:

```python
    from training.train import set_seed
    from config import SEED
    set_seed(SEED)
```

- [ ] **Step 4: Verify reproducibility with a one-liner**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "
import torch
from training.train import set_seed
set_seed(42); a = torch.randn(3)
set_seed(42); b = torch.randn(3)
print(a.tolist()); print(b.tolist())
assert torch.equal(a, b)
print('SEEDS OK')
"
```
Expected: two identical 3-element tensors printed, then `SEEDS OK`.

- [ ] **Step 5: Commit**

```bash
git add training/train.py main.py
git commit -m "training: add set_seed() helper and call it from cross_validate + main"
```

---

## Task 11: Modify `cross_validate(...)` — per-fold standardization + return preds/targets

**Files:**
- Modify: `training/train.py`

- [ ] **Step 1: Add a `standardize_node_features` helper to `training/train.py`**

Insert this function below `set_seed`:

```python
import copy


def standardize_node_features(train_graphs, val_graphs):
    """
    Z-score node features (Data.x) per column using *training-fold* statistics only.
    Returns new lists with standardized x; original Data objects untouched.
    """
    # Stack all training x → (N_train * 189, n_features)
    train_x = torch.cat([g.x for g in train_graphs], dim=0)
    mean = train_x.mean(dim=0)        # shape (n_features,)
    std = train_x.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)  # avoid div-by-zero

    def _apply(graphs):
        out = []
        for g in graphs:
            g2 = copy.copy(g)         # shallow copy
            g2.x = (g.x - mean) / std
            out.append(g2)
        return out

    return _apply(train_graphs), _apply(val_graphs)
```

- [ ] **Step 2: Replace the body of `cross_validate(...)` with a version that standardizes per fold and collects preds/targets**

Replace the entire `cross_validate` function with:

```python
def cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS):
    """Run k-fold cross-validation. Returns (fold_metrics, all_preds, all_targets)."""
    set_seed(SEED)
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
                                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            if epoch % 50 == 0:
                preds, targets = evaluate(model, val_loader, device)
                metrics = compute_metrics(targets, preds)
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
                      f"r: {metrics['pearson_r']:.3f}")

        preds, targets = evaluate(model, val_loader, device)
        metrics = compute_metrics(targets, preds)
        fold_metrics.append(metrics)
        all_preds.append(preds)
        all_targets.append(targets)
        print(f"  Final → MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
              f"r: {metrics['pearson_r']:.3f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return fold_metrics, all_preds, all_targets
```

- [ ] **Step 3: Write a quick standardization test**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "
import torch
from torch_geometric.data import Data
from training.train import standardize_node_features

# Build 3 toy training graphs with x of shape (4, 2) each
def mk(x): return Data(x=torch.tensor(x, dtype=torch.float))
train = [mk([[0.5, 1e-3], [0.4, 9e-4], [0.6, 1.1e-3], [0.3, 8e-4]]) for _ in range(3)]
val   = [mk([[0.5, 1e-3], [0.4, 9e-4]])]
ts, vs = standardize_node_features(train, val)
stacked = torch.cat([g.x for g in ts], dim=0)
print('train mean:', stacked.mean(dim=0).tolist())
print('train std :', stacked.std(dim=0).tolist())
assert abs(stacked.mean(dim=0)[0]) < 1e-5
assert abs(stacked.std(dim=0)[0]  - 1.0) < 1e-5
print('STANDARDIZE OK')
"
```
Expected: train-fold mean is ≈ 0 and std is ≈ 1 per feature; prints `STANDARDIZE OK`.

- [ ] **Step 4: Commit**

```bash
git add training/train.py
git commit -m "training: per-fold standardize node features (training-fold stats only)

- New standardize_node_features(train, val) helper: z-scores Data.x per column
  using training-fold mean/std, avoiding leakage.
- cross_validate() now applies it inside each fold and returns
  (fold_metrics, all_preds, all_targets) so callers can plot
  concatenated predictions across folds.

Resolves the FA (~[0.2, 0.7]) vs. MD (~[5e-4, 1.5e-3]) ~1000x scale gap
that would otherwise dominate the GCN's first-layer gradients."
```

---

## Task 12: Update `main.py` — drive ridge + GCN, save scatter plots and metrics JSON

**Files:**
- Modify: `baselines/ridge_regression.py` (return preds/targets in addition to metrics)
- Modify: `main.py`
- Modify: `training/evaluate.py` (one-line headless-mode tweak)

- [ ] **Step 1: Refactor `baselines/ridge_regression.py` to return preds/targets**

Edit `baselines/ridge_regression.py`. Find the function body of `run_ridge_baseline` and replace it with:

```python
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

    import numpy as _np
    return fold_metrics, _np.concatenate(all_preds), _np.concatenate(all_targets)
```

The only changes vs. the existing implementation: collect `preds` and `y[val_idx]` per fold, return them concatenated.

- [ ] **Step 2: Replace `main.py` end-to-end**

Overwrite `main.py` with:

```python
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
```

- [ ] **Step 3: Make `plot_predictions` non-blocking when running headlessly**

Edit `training/evaluate.py`. Find:
```python
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

Replace with:
```python
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
```

(`plt.show()` blocks in non-interactive runs. We always save to disk; the CLI doesn't need to display.)

- [ ] **Step 4: Verify imports parse**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "import main; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add baselines/ridge_regression.py main.py training/evaluate.py
git commit -m "main: drive ridge + GCN, save scatter plots and metrics JSON

- ridge_regression: return concatenated preds/targets in addition to metrics.
- main.py: ridge baseline always runs (cheap, required by 2025 npj AI critique);
  GCN cross-validation produces preds per fold.
- Per-target outputs:
    results/scatter_<target>_ridge.png
    results/scatter_<target>_<model>.png
    results/metrics_<target>.json
- evaluate.plot_predictions: replace plt.show() with plt.close(fig)
  so headless runs don't block."
```

---

## Task 13: Smoke-test the ridge baseline on `MoCA_T`

**Files:** none (verification step). Generates `results/scatter_moca_t_ridge.png` and `results/metrics_moca_t.json`.

- [ ] **Step 1: Run the ridge baseline only**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python main.py --target MoCA_T --baseline_only`

Expected: per-fold lines like `Fold 1: MAE=2.xxx  R²=0.xxx  r=0.xxx`, then a summary block, then `Wrote .../results/scatter_moca_t_ridge.png` and `Wrote .../results/metrics_moca_t.json`. Total runtime: under 2 minutes.

- [ ] **Step 2: Inspect the JSON**

Run: `cat /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/metrics_moca_t.json | python -m json.tool | head -30`
Expected: valid JSON with keys `target`, `node_features`, `seed`, `ridge.per_fold` (list of 5), `ridge.mean` (with `mae`, `r2`, `pearson_r`), and `gcn: null`.

- [ ] **Step 3: Re-run to verify reproducibility**

Run: `python main.py --target MoCA_T --baseline_only` (a second time).
Expected: per-fold metrics are bit-identical to the first run (proves seeds work for the ridge path).

- [ ] **Step 4: Commit (optional — if metrics file ends up tracked, untrack it)**

The metrics JSON is gitignored per Task 2. No commit needed unless investigation revealed a bug, in which case the bug-fix commit goes here.

---

## Task 14: Run GCN cross-validation on `MoCA_T`

**Files:** none (verification step). Generates `results/scatter_moca_t_gcn.png` and updates `results/metrics_moca_t.json`.

- [ ] **Step 1: Run the GCN cross-validation**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && time python main.py --target MoCA_T --model gcn`

Expected: dataset loads from cache (created in Task 9 verification), then 5 fold blocks each with periodic `Epoch  50 / 100 / 150 / 200` lines, then a final summary. Total runtime: 10–60 minutes on CPU depending on machine.

**Stop condition:** if elapsed time exceeds 30 minutes and not yet on fold 4, kill the run, edit `config.py` to set `EPOCHS = 100`, and rerun. Note this in commit message at the end of Task 15.

- [ ] **Step 2: Inspect the scatter plot and JSON**

Run: `ls -la /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/scatter_moca_t_gcn.png /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/metrics_moca_t.json`
Expected: both files exist, scatter > 10KB.

Open `metrics_moca_t.json` and confirm both `ridge` and `gcn` sections are populated with 5 folds each.

- [ ] **Step 3: Re-run to verify reproducibility**

Run: `python main.py --target MoCA_T --model gcn` (second time).
Expected: per-fold MAE/R²/r values bit-identical to first run.

- [ ] **Step 4: No commit (artifacts gitignored). If the run revealed a bug, fix and commit the fix.**

---

## Task 15: Run ridge + GCN for `PTT_Average`

**Files:** none (verification step). Generates `results/scatter_ptt_average_ridge.png`, `results/scatter_ptt_average_gcn.png`, and `results/metrics_ptt_average.json`.

- [ ] **Step 1: Run the GCN cross-validation for PTT_Average**

Run: `cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && time python main.py --target PTT_Average --model gcn`

Expected: dataset builds from .mat (different cache key from MoCA — first run for this target), then 5 folds. ~277 subjects after dropping 28 with missing PTT. Total runtime similar to Task 14.

- [ ] **Step 2: Inspect outputs**

Run: `ls -la /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/scatter_ptt_average_gcn.png /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind/results/metrics_ptt_average.json`
Expected: both files exist.

- [ ] **Step 3: Print final summary across both targets**

Run:
```bash
cd /Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN/GraphMind && python -c "
import json, os
RES = 'results'
for fn in ['metrics_moca_t.json', 'metrics_ptt_average.json']:
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        print(f'{fn}: MISSING'); continue
    d = json.load(open(p))
    print(f'\n=== {d[\"target\"]} (n_folds={d[\"n_folds\"]}, features={d[\"node_features\"]}) ===')
    for method in ['ridge', 'gcn']:
        m = d.get(method)
        if m is None:
            print(f'  {method}: not run'); continue
        mn = m['mean']
        print(f'  {method:5s}: MAE={mn[\"mae\"]:.3f}  R^2={mn[\"r2\"]:.3f}  r={mn[\"pearson_r\"]:.3f}')
"
```
Expected: a 4-row summary (2 targets × 2 methods) with mean MAE, R², and Pearson r.

- [ ] **Step 4: Final commit — sprint completion marker**

If `EPOCHS` was reduced in Task 14, include that change here:
```bash
git add config.py 2>/dev/null || true
git commit --allow-empty -m "milestone: first end-to-end GraphMind run complete

Outputs (gitignored):
  results/scatter_moca_t_{ridge,gcn}.png
  results/scatter_ptt_average_{ridge,gcn}.png
  results/metrics_{moca_t,ptt_average}.json

Both targets ran 5-fold CV with per-fold node-feature standardization
and a ridge baseline for comparison. Reproducibility verified via
seeded reruns producing bit-identical metrics."
```

---

## Sprint completion checklist

After all 15 tasks, the following are true (per the spec's success criteria):

- [ ] `python scripts/smoke_test.py` prints `SMOKE TEST PASSED`.
- [ ] `python main.py --target MoCA_T --baseline_only` produces ridge metrics JSON.
- [ ] `python main.py --target MoCA_T --model gcn` produces GCN metrics + scatter PNG.
- [ ] Same for `PTT_Average`.
- [ ] Two reruns of the same command produce bit-identical numbers.
- [ ] Cached dataset loads in < 10 seconds on second invocation.

If any of the above is unmet, do not declare the sprint complete — investigate.
