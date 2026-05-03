"""
gat_regression.py — Graph Attention Network for regression.

Attention weights per edge provide interpretability:
which connections contribute most to the prediction?
Use with GNNExplainer (interpretation/explain.py) for full interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import HIDDEN_DIM, NUM_LAYERS, DROPOUT, HEADS


class GATRegression(nn.Module):
    def __init__(self, num_node_features, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GATConv(num_node_features, hidden_dim,
                                  heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim,
                                      heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)
