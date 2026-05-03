"""
explain.py — Interpretability tools for identifying important brain regions and connections.

Methods:
    1. GAT attention weights  → which edges are attended to most
    2. GNNExplainer           → minimal subgraph important for a prediction
    3. Node importance scores → aggregate edge importances per region
"""

import torch
import numpy as np


def extract_gat_attention(model, data, device):
    """
    Extract attention weights from a trained GAT model for a single graph.
    Returns edge_index and corresponding attention weights.
    """
    model.eval()
    data = data.to(device)
    attention_weights = []

    def hook(module, input, output):
        # GATConv returns (out, (edge_index, alpha))
        if isinstance(output, tuple):
            attention_weights.append(output[1][1].detach().cpu())

    hooks = [conv.register_forward_hook(hook) for conv in model.convs]
    with torch.no_grad():
        _ = model(data.x, data.edge_index, data.edge_attr,
                  torch.zeros(data.num_nodes, dtype=torch.long, device=device))
    for h in hooks:
        h.remove()

    return data.edge_index.cpu(), attention_weights


def node_importance_from_edges(edge_index, edge_weights, n_nodes):
    """
    Aggregate edge-level importance to node-level by summing weights of incident edges.
    Returns array of shape [n_nodes].
    """
    node_scores = np.zeros(n_nodes)
    src, dst = edge_index.numpy()
    for i, w in enumerate(edge_weights):
        node_scores[src[i]] += abs(w)
        node_scores[dst[i]] += abs(w)
    return node_scores


def gnnexplainer_explain(model, data, device, epochs=200):
    """
    Run GNNExplainer on a single graph to identify important edges and features.
    Requires torch_geometric.explain.
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    data = data.to(device)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="graph", return_type="raw"),
    )
    explanation = explainer(
        data.x, data.edge_index,
        edge_attr=data.edge_attr,
        batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    )
    return explanation
