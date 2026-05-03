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
    assert torch.isfinite(out).all(), f"model output contains non-finite values: {out}"
    pred = float(out.item())
    actual = float(data.y.item())

    print(f"  edges={n_edges}  pred={pred:.3f}  actual={actual:.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
