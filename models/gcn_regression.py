"""
gcn_regression.py - Graph regression network with signed-weight message passing.

Why GraphConv (not GCNConv):
- GCNConv applies symmetric normalization (D^(-1/2) · A · D^(-1/2)) which
  mathematically requires non-negative edge weights. The Pearson correlations
  in data.edge_attr are signed (typically -0.3 to +0.9 in our top-15% edges).
- GraphConv computes x_i' = W1·x_i + W2·sum_j(e_ij · x_j): a weighted sum of
  neighbour features with a separate self-transform, no symmetric normalization.
  Signed weights are mathematically valid here — positive edges add the
  neighbour's features, negative edges subtract them, scaled by magnitude.

Architecture:
    GraphConv layers (signed-weight message passing) → BatchNorm → ReLU → Dropout
    → Global mean pool → Linear regression head (scalar output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from config import HIDDEN_DIM, NUM_LAYERS, DROPOUT


class GCNRegression(nn.Module):
    def __init__(self, num_node_features, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GraphConv(num_node_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is shape [num_edges, 1]; GraphConv wants edge_weight as [num_edges]
        edge_weight = edge_attr.squeeze(-1)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)   # graph-level representation
        return self.head(x).squeeze(-1)  # scalar prediction per graph
