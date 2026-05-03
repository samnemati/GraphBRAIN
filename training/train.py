"""
train.py — Training loop and k-fold cross-validation.
"""

import copy
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from config import N_FOLDS, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, SEED
from training.evaluate import compute_metrics


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.append(pred.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def cross_validate(model_class, dataset, model_kwargs, device, n_folds=N_FOLDS,
                   lr=None, patience=20):
    """Run k-fold cross-validation with per-fold node-feature standardization
    and early stopping on validation MAE.

    Args:
        lr: optional learning rate; falls back to config.LEARNING_RATE.
        patience: epochs without val-MAE improvement before stopping.

    Returns: (fold_metrics, all_preds, all_targets)
    """
    set_seed(SEED)
    if lr is None:
        lr = LEARNING_RATE

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
                                     lr=lr, weight_decay=WEIGHT_DECAY)

        best_mae = float("inf")
        best_state = None
        best_epoch = 0
        epochs_since_best = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            preds, targets = evaluate(model, val_loader, device)
            metrics = compute_metrics(targets, preds)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
                      f"r: {metrics['pearson_r']:.3f}")

            if epochs_since_best >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"best val MAE {best_mae:.3f}).")
                break

        # Restore best-MAE weights for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)

        preds, targets = evaluate(model, val_loader, device)
        metrics = compute_metrics(targets, preds)
        fold_metrics.append(metrics)
        all_preds.append(preds)
        all_targets.append(targets)
        print(f"  Final (best-weights) → MAE: {metrics['mae']:.3f} | "
              f"R²: {metrics['r2']:.3f} | r: {metrics['pearson_r']:.3f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return fold_metrics, all_preds, all_targets
