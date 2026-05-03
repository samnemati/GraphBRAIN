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
