"""
graph_builder.py — Convert per-subject connectivity + features into PyG Data objects.

Graph definition:
    Nodes  : 189 JHU atlas brain regions
    Edges  : Top THRESHOLD_PERCENT strongest functional connections (proportional threshold)
    X      : Node feature matrix [189 × n_features]
    y      : Continuous behavioral score (regression target)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from config import THRESHOLD_PERCENT, N_REGIONS


def threshold_connectivity(connectivity, threshold_percent=THRESHOLD_PERCENT):
    """
    Apply proportional threshold: keep top k% connections.
    Returns binary mask and thresholded weight matrix.
    """
    # Use upper triangle only (symmetric matrix)
    triu_indices = np.triu_indices(N_REGIONS, k=1)
    weights = connectivity[triu_indices]

    # Determine threshold value
    cutoff = np.percentile(np.abs(weights), (1 - threshold_percent) * 100)
    mask = np.abs(weights) >= cutoff

    src = triu_indices[0][mask]
    dst = triu_indices[1][mask]
    edge_weights = weights[mask]

    # Make undirected (add both directions)
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ], axis=0)
    edge_attr = np.concatenate([edge_weights, edge_weights])

    return edge_index, edge_attr


def build_graph(subject_dict, target_col):
    """
    Build a PyG Data object for one subject.

    Args:
        subject_dict (dict): output from loader.load_all_subjects()
        target_col (str): column name for regression target (e.g. 'MoCA_T')

    Returns:
        torch_geometric.data.Data or None (if target is NaN)
    """
    behavioral = subject_dict["behavioral"]
    y_val = behavioral[target_col]

    if pd.isna(y_val):
        return None

    # Node features — skip subjects with NaN/Inf (would poison per-fold standardization)
    x = torch.tensor(subject_dict["node_features"], dtype=torch.float)
    if not torch.isfinite(x).all():
        print(f"  Warning: skipping {subject_dict['subject_id']}: non-finite values in node features")
        return None

    # Edges
    edge_index, edge_attr = threshold_connectivity(subject_dict["connectivity"])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    if not torch.isfinite(edge_attr).all():
        print(f"  Warning: skipping {subject_dict['subject_id']}: non-finite values in edge weights")
        return None

    # Target
    y = torch.tensor([float(y_val)], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        subject_id=subject_dict["subject_id"],
    )
