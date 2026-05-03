"""
loader.py — Load subject .mat files and extract connectivity matrices + node features.

Each subject .mat file contains (among other things):
    rest_jhu['r']      : 189×189 resting-state FC matrix (Pearson r)
    rest_jhu['label']  : 189 ROI names
    fa_jhu['mean']     : FA per region
    md_jhu['mean']     : MD per region
    vbm_gm_jhu['mean'] : gray matter volume per region
    vbm_wm_jhu['mean'] : white matter volume per region
    palf_jhu['mean']   : amplitude of low-freq fluctuations per region
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from config import MAT_DIR_S1, BEHAVIORAL_FILE, NODE_FEATURES, ATLAS


def load_behavioral_data(filepath=BEHAVIORAL_FILE):
    """Load and return the behavioral DataFrame, indexed by Study ID."""
    df = pd.read_excel(filepath)
    df["Study ID"] = df["Study ID"].astype(str).str.strip()
    df = df.set_index("Study ID")
    return df


def load_subject_mat(filepath):
    """
    Load a single subject .mat file and extract graph components.

    Returns:
        connectivity (np.ndarray): 189×189 FC matrix from rest_jhu['r']
        node_features (np.ndarray): 189 × len(NODE_FEATURES) matrix
        roi_labels (list): list of 189 ROI name strings
    """
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    # --- Connectivity matrix ---
    rest = mat[f"rest_{ATLAS}"]
    connectivity = rest.r                      # shape: (189, 189)
    roi_labels = list(rest.label.flat)

    # --- Node features ---
    feature_list = []
    for feat_name in NODE_FEATURES:
        key = f"{feat_name}_{ATLAS}"
        feat_data = np.asarray(mat[key].mean)  # shape: (189,) or (1, 189)
        feature_list.append(feat_data.flatten())

    node_features = np.stack(feature_list, axis=1)  # shape: (189, n_features)

    return connectivity, node_features, roi_labels


def load_all_subjects(mat_dir=MAT_DIR_S1, behavioral_df=None):
    """
    Load all subjects with both brain and behavioral data.

    Returns:
        subject_data (list of dicts): one dict per subject with keys:
            'subject_id', 'connectivity', 'node_features', 'roi_labels', 'behavioral'
    """
    if behavioral_df is None:
        behavioral_df = load_behavioral_data()

    mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mat")])
    subject_data = []

    for fname in tqdm(mat_files, desc="Loading subjects"):
        subject_id = fname.replace(".mat", "")

        # Skip if no behavioral data
        if subject_id not in behavioral_df.index:
            continue

        filepath = os.path.join(mat_dir, fname)
        try:
            connectivity, node_features, roi_labels = load_subject_mat(filepath)
            subject_data.append({
                "subject_id": subject_id,
                "connectivity": connectivity,
                "node_features": node_features,
                "roi_labels": roi_labels,
                "behavioral": behavioral_df.loc[subject_id],
            })
        except Exception as e:
            print(f"  Warning: failed to load {fname}: {e}")

    print(f"Loaded {len(subject_data)} subjects successfully.")
    return subject_data
