# GraphMind

Graph Neural Network regression pipeline for predicting behavioral outcomes
(cognitive and hearing scores) from resting-state functional brain connectivity
in the ABC Healthy Aging cohort (ages 20–80).

## Overview

This project uses PyTorch Geometric (PyG) to build a brain graph per subject
(189 JHU atlas regions as nodes, resting-state functional connectivity as edges)
and trains GCN/GAT models to predict:

- **MoCA_T** — Montreal Cognitive Assessment score
- **PTT_Average** — Pure Tone Threshold (hearing)
- *(planned)* WIN threshold, Age

## Project Structure

```
GraphMind/
├── config.py               ← paths and hyperparameters
├── data/                   ← data loading and graph construction
├── models/                 ← GCN and GAT regression models
├── training/               ← training loop and evaluation
├── interpretation/         ← GNNExplainer and attention weight analysis
├── baselines/              ← ridge regression comparison
├── notebooks/              ← exploration and visualization
└── results/                ← saved model outputs
```

## Setup

See `requirements.txt` for dependencies. Install PyTorch and PyTorch Geometric
following the official guides before running `pip install -r requirements.txt`.

## Full Project Context

See `CLAUDE.md` for complete project briefing including data structure,
design decisions, and literature review.
