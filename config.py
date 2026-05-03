"""
config.py — Global configuration for GraphMind pipeline.
All paths, hyperparameters, and toggles live here.
"""

import os

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/snemati/Documents/ABC_BrainAge/Working_Study/GNN"
MAT_DIR_S1 = os.path.join(DATA_ROOT, "Brain_Data", "ABC_MATfiles_session1")
MAT_DIR_S2 = os.path.join(DATA_ROOT, "Brain_Data", "ABC_MATfiles_session2")
BEHAVIORAL_FILE = os.path.join(DATA_ROOT, "Behavioral_Data", "ABC_PTT_WIN_MoCA.xlsx")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Graph construction ───────────────────────────────────────────────────────
ATLAS = "jhu"                    # atlas to use (jhu = 189 regions)
N_REGIONS = 189                  # number of JHU atlas regions
EDGE_TYPE = "rest"               # 'rest' = resting-state FC, 'dti' = structural
THRESHOLD_PERCENT = 0.15         # keep top 15% connections (proportional threshold)

# Node features to include (must be present in .mat as {feat}_jhu)
NODE_FEATURES = ["fa", "md"]

# ── Regression targets ────────────────────────────────────────────────────────
TARGET = "MoCA_T"               # 'MoCA_T', 'PTT_Average', 'WIN_Threshold_Average', 'Age'

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_TYPE = "gcn"              # 'gcn' or 'gat'
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
HEADS = 4                       # GAT only: number of attention heads

# ── Training ─────────────────────────────────────────────────────────────────
N_FOLDS = 5                     # k-fold cross-validation
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
SEED = 42
