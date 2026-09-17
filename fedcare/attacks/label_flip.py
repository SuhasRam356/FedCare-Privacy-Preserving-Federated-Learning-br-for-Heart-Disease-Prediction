"""
fedcare/attacks/label_flip.py - Label Flipping Attack Implementation.

Simulates a malicious healthcare node (e.g., compromised Hospital 4) that systematically
inverts or corrupts binary diagnosis labels (0 -> 1, 1 -> 0) prior to local model training.
This creates targeted classification poisoning and drives the global model toward
misclassifying heart-disease cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def apply_label_flip(
    df: pd.DataFrame,
    flip_rate: float = 1.0,
    source_class: int | None = None,
    target_class: int | None = None,
    target_col: str = "target",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply label-flipping corruption to a hospital DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original client dataset.
    flip_rate : float, default=1.0
        Fraction of samples whose labels will be flipped (0.0 to 1.0).
    source_class : int | None, optional
        If specified, only flip samples with this class label. If None, invert binary (0 <-> 1).
    target_class : int | None, optional
        Target class to flip into if source_class is specified.
    target_col : str, default="target"
        Name of the target label column.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A modified copy of the dataset containing poisoned labels.
    """
    df_poisoned = df.copy()
    if target_col not in df_poisoned.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    rng = np.random.default_rng(seed)
    n_samples = len(df_poisoned)

    if source_class is not None and target_class is not None:
        mask = df_poisoned[target_col] == source_class
        indices_to_flip = df_poisoned[mask].index.to_numpy()
        flip_count = int(len(indices_to_flip) * flip_rate)
        if flip_count > 0:
            selected_indices = rng.choice(indices_to_flip, size=flip_count, replace=False)
            df_poisoned.loc[selected_indices, target_col] = target_class
    else:
        # Full binary label inversion: 0 -> 1 and 1 -> 0
        flip_mask = rng.random(n_samples) < flip_rate
        original_labels = df_poisoned[target_col].to_numpy()
        df_poisoned.loc[flip_mask, target_col] = 1 - original_labels[flip_mask]

    return df_poisoned


def poison_dataloader_labels(
    loader: DataLoader,
    flip_rate: float = 1.0,
    seed: int = 42,
    shuffle: bool | None = None,
) -> DataLoader:
    """
    Convert an existing PyTorch DataLoader into a poisoned DataLoader with flipped labels.
    """
    all_x = []
    all_y = []
    for batch_x, batch_y in loader:
        all_x.append(batch_x)
        all_y.append(batch_y)

    x_tensor = torch.cat(all_x, dim=0)
    y_tensor = torch.cat(all_y, dim=0)

    # Invert binary labels
    rng = np.random.default_rng(seed)
    n = len(y_tensor)
    mask = rng.random(n) < flip_rate
    y_poisoned = y_tensor.clone()
    y_poisoned[mask] = 1 - y_poisoned[mask]

    if shuffle is None:
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)

    dataset = TensorDataset(x_tensor, y_poisoned)
    return DataLoader(
        dataset,
        batch_size=loader.batch_size or 32,
        shuffle=shuffle,
    )
