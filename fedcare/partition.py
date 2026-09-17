"""
fedcare.partition - Dataset partitioning strategies for Non-IID heterogeneity studies.

Supports three partitioning regimes:
1. Hospital-Native (Real Skew):
   Uses the genuine 6-hospital dataset splits, preserving real-world clinical
   demographic and prevalence variations (4.8% to 46.3% disease prevalence).
2. IID (Independent and Identically Distributed):
   Random uniform distribution of the pooled dataset across N clients with
   matching class distributions.
3. Synthetic Dirichlet Heterogeneity:
   Dirichlet distribution Dir(alpha) over class labels with concentration
   parameter alpha in {0.1, 0.5, 1.0}. Lower alpha induces extreme label distribution skew.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from fedcare.task import (
    BATCH_SIZE,
    DATA_DIR,
    HeartDiseaseDataset,
    RANDOM_STATE,
    TEST_SPLIT,
    load_data as load_hospital_native,
)

logger = logging.getLogger("fedcare.partition")


def partition_iid(
    num_partitions: int = 6,
    batch_size: int = BATCH_SIZE,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_STATE,
) -> dict[int, tuple[DataLoader, DataLoader, StandardScaler]]:
    """
    Split the pooled 12,000-patient dataset into uniform, balanced IID partitions.

    Parameters
    ----------
    num_partitions : int
        Number of client partitions to create (default: 6).
    batch_size : int
        Mini-batch size for DataLoader.
    test_split : float
        Hold-out evaluation split fraction (default: 0.2).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    dict[int, tuple[DataLoader, DataLoader, StandardScaler]]
        Dictionary mapping partition_id (1 to num_partitions) to
        (train_loader, test_loader, scaler).
    """
    combined_csv = DATA_DIR / "combined.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"Missing {combined_csv}. Run prepare_data.py first.")

    df = pd.read_csv(combined_csv)
    rng = np.random.default_rng(seed)

    # Stratified shuffle across all rows
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    chunk_size = len(df_shuffled) // num_partitions
    chunks = [
        df_shuffled.iloc[i * chunk_size : (i + 1) * chunk_size].copy().reset_index(drop=True)
        for i in range(num_partitions)
    ]

    partitions: dict[int, tuple[DataLoader, DataLoader, StandardScaler]] = {}

    for idx, chunk_df in enumerate(chunks, start=1):
        X = chunk_df.drop(columns=["target"]).values
        y = chunk_df["target"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=seed, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_ds = HeartDiseaseDataset(X_train, y_train)
        test_ds = HeartDiseaseDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        partitions[idx] = (train_loader, test_loader, scaler)

    return partitions


def partition_dirichlet(
    alpha: float,
    num_partitions: int = 6,
    batch_size: int = BATCH_SIZE,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_STATE,
) -> dict[int, tuple[DataLoader, DataLoader, StandardScaler]]:
    """
    Partition the pooled dataset using a Dirichlet distribution Dir(alpha) over class labels.

    A lower alpha (e.g. 0.1) results in extreme label heterogeneity where clients
    receive predominantly one class. A higher alpha (e.g. 1.0) creates mild skew.

    Parameters
    ----------
    alpha : float
        Dirichlet concentration parameter (e.g. 0.1, 0.5, 1.0).
    num_partitions : int
        Number of client partitions (default: 6).
    batch_size : int
        Mini-batch size.
    test_split : float
        Hold-out test fraction.
    seed : int
        Random seed.

    Returns
    -------
    dict[int, tuple[DataLoader, DataLoader, StandardScaler]]
        Mapping of partition_id (1..N) to (train_loader, test_loader, scaler).
    """
    combined_csv = DATA_DIR / "combined.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"Missing {combined_csv}. Run prepare_data.py first.")

    df = pd.read_csv(combined_csv)
    rng = np.random.default_rng(seed)

    # Separate class indices
    y = df["target"].values.astype(int)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(num_partitions)]
    min_per_class = 10

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)

        # Ensure every client receives at least min_per_class samples for valid evaluation
        for client_idx in range(num_partitions):
            start = client_idx * min_per_class
            end = (client_idx + 1) * min_per_class
            client_indices[client_idx].extend(idx_c[start:end].tolist())

        remaining_idx = idx_c[num_partitions * min_per_class :]

        # Draw proportions from Dirichlet distribution for the remaining samples
        proportions = rng.dirichlet(np.repeat(alpha, num_partitions))
        proportions = np.cumsum(proportions) / np.sum(proportions)
        split_points = (proportions[:-1] * len(remaining_idx)).astype(int)
        class_splits = np.split(remaining_idx, split_points)

        for client_idx, split_idx in enumerate(class_splits):
            client_indices[client_idx].extend(split_idx.tolist())

    partitions: dict[int, tuple[DataLoader, DataLoader, StandardScaler]] = {}

    for client_idx, indices in enumerate(client_indices, start=1):
        client_df = df.iloc[indices].reset_index(drop=True)
        X = client_df.drop(columns=["target"]).values
        y_client = client_df["target"].values.astype(int)

        # Check if both classes exist and have sufficient samples for stratification
        unique_classes, counts = np.unique(y_client, return_counts=True)
        strat = (
            y_client
            if (len(unique_classes) > 1 and np.min(counts) >= 2)
            else None
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_client, test_size=test_split, random_state=seed, stratify=strat
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_ds = HeartDiseaseDataset(X_train, y_train)
        test_ds = HeartDiseaseDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        partitions[client_idx] = (train_loader, test_loader, scaler)

    return partitions


def get_partition_data(
    partition_type: str = "hospital_native",
    partition_id: int = 1,
    alpha: Optional[float] = None,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Unified entrypoint to fetch a specific partition's DataLoaders.

    Parameters
    ----------
    partition_type : str
        One of {'hospital_native', 'iid', 'dirichlet'}.
    partition_id : int
        Hospital or client index (1 through 6).
    alpha : float, optional
        Concentration parameter for Dirichlet partitioning (required if partition_type='dirichlet').
    batch_size : int
        Batch size.
    seed : int
        Random seed.

    Returns
    -------
    tuple[DataLoader, DataLoader, StandardScaler]
        (train_loader, test_loader, scaler)
    """
    if partition_type == "hospital_native":
        return load_hospital_native(partition_id=partition_id, batch_size=batch_size)

    elif partition_type == "iid":
        partitions = partition_iid(num_partitions=6, batch_size=batch_size, seed=seed)
        return partitions[partition_id]

    elif partition_type == "dirichlet":
        if alpha is None:
            raise ValueError("alpha must be specified for Dirichlet partitioning.")
        partitions = partition_dirichlet(
            alpha=alpha, num_partitions=6, batch_size=batch_size, seed=seed
        )
        return partitions[partition_id]

    else:
        raise ValueError(
            f"Unknown partition_type '{partition_type}'. "
            "Choose from 'hospital_native', 'iid', or 'dirichlet'."
        )
