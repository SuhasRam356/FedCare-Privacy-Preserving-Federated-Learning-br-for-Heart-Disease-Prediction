"""
fedcare/task.py – Core building blocks for FedCare.

Contains:
    HeartDiseaseDataset  – PyTorch Dataset for tabular CSV data
    Net                  – MLP model (12 → 64 → 32 → 2)
    load_data()          – loads centralized or per-hospital data
    train()              – one-epoch training loop
    test()               – evaluation loop returning loss, accuracy, AUC
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "heart"
NUM_FEATURES = 13
NUM_CLASSES = 2
BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_STATE = 42


# ── Dataset ───────────────────────────────────────────────────────────
class HeartDiseaseDataset(Dataset):
    """PyTorch Dataset wrapping a NumPy feature matrix and label vector."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    Simple MLP for binary classification on 13-feature tabular data.

    Features: age, resting_bp, cholesterol, max_heart_rate, bmi, glucose, 
              sex, smoker, diabetes_history, family_history, 
              cp_atypical_angina, cp_non_anginal, cp_typical_angina

    Architecture:
        Input (13) -> Linear(64) -> ReLU -> Dropout(0.3)
                   -> Linear(32) -> ReLU -> Dropout(0.3)
                   -> Linear(2)   (logits for CrossEntropyLoss)
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Data loading ──────────────────────────────────────────────────────
def load_data(
    partition_id: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    test_split: float = TEST_SPLIT,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Load heart-disease CSV data and return train/test DataLoaders.

    Args:
        partition_id: If ``None``, loads ``combined.csv`` (centralized).
                      If an int (1-6), loads ``hospital_{partition_id}.csv``.
        batch_size:   Mini-batch size for the DataLoaders.
        test_split:   Fraction of data reserved for testing.

    Returns:
        (train_loader, test_loader, scaler)
    """
    if partition_id is not None:
        csv_path = DATA_DIR / f"hospital_{partition_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                "Ensure synthetic datasets have been generated."
            )
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values.astype(int)
        
        # Train / test split (stratified to preserve class ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=RANDOM_STATE, stratify=y,
        )
    else:
        # [CRITICAL FIX]: To prevent data leakage, the global/centralized dataset
        # MUST be assembled by concatenating the EXACT SAME splits the hospitals use.
        # Otherwise, the global test set would randomly include rows the hospitals trained on.
        X_tr_list, X_te_list, y_tr_list, y_te_list = [], [], [], []
        for i in range(1, 7):
            csv_path = DATA_DIR / f"hospital_{i}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            X = df.drop(columns=["target"]).values
            y = df["target"].values.astype(int)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_split, random_state=RANDOM_STATE, stratify=y,
            )
            X_tr_list.append(X_tr)
            X_te_list.append(X_te)
            y_tr_list.append(y_tr)
            y_te_list.append(y_te)
            
        X_train = np.vstack(X_tr_list)
        X_test = np.vstack(X_te_list)
        y_train = np.concatenate(y_tr_list)
        y_test = np.concatenate(y_te_list)

    # StandardScaler fitted on training data only (no data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = HeartDiseaseDataset(X_train, y_train)
    test_ds = HeartDiseaseDataset(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        worker_init_fn=seed_worker, generator=g
    )

    return train_loader, test_loader, scaler


# ── Training ──────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    mu: float = 0.0,
    global_model: Optional[nn.Module] = None,
    c_local: Optional[list[torch.Tensor]] = None,
    c_global: Optional[list[torch.Tensor]] = None,
    use_opacus: bool = False,
    dp_max_grad_norm: float = 1.0,
    dp_noise_multiplier: float = 1.0,
) -> tuple[float, int]:
    """
    Train ``model`` for ``epochs`` epochs on ``train_loader`` with optional FedProx proximal term
    and optional rigorous Differential Privacy (DP-SGD) via Opacus.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = None
    if use_opacus:
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
            )
        except ImportError:
            import logging
            logging.getLogger("fedcare.task").error("Opacus is not installed. Run 'pip install opacus' to use rigorous DP-SGD.")
            use_opacus = False

    epoch_loss = 0.0
    n_batches = 0
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            # FedProx proximal regularization
            if mu > 0.0 and global_model is not None:
                proximal_term = torch.tensor(0.0, device=device)
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += torch.sum((w - w_t.to(device)) ** 2)
                loss = loss + (mu / 2.0) * proximal_term

            loss.backward()

            # SCAFFOLD Control Variate Adjustment (Skip if Opacus used, as DPOptimizer handles grads differently)
            if c_local is not None and c_global is not None and not use_opacus:
                for param, c_l, c_g in zip(model.parameters(), c_local, c_global):
                    if param.grad is not None:
                        param.grad.data += (c_g.to(device) - c_l.to(device))

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1
            
    if use_opacus and privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        import logging
        logging.getLogger("fedcare.task").info(f"DP-SGD Epochs complete. Privacy guarantee: ε = {epsilon:.2f}, δ = 1e-5")
        # Remove opacus hooks before returning to avoid issues with state_dict
        model.remove_hooks()

    avg_loss = epoch_loss / max(n_batches, 1)
    return avg_loss, total_steps


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """
    Evaluate ``model`` on ``test_loader``.

    Returns:
        Dictionary with keys: ``loss``, ``accuracy``, ``auc``.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    # AUC – handle edge case where only one class is present in the batch
    all_probs_np = np.concatenate(all_probs)
    all_labels_np = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        auc = 0.0  # only one class present

    return {"loss": avg_loss, "accuracy": accuracy, "auc": auc}

# ── Personalized Federated Learning Architectures ────────────────────────

class FedPerNet(nn.Module):
    """
    FedPer (Base + Head split).
    The base layers (feature extractor) are federated.
    The head layers (classifier) are kept strictly local.
    """
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(32, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return self.head(features)


class FedBNNet(nn.Module):
    """
    FedBN (Batch Normalization local).
    All layers except BatchNorm layers are federated.
    BatchNorm statistics (mean, var) and affine parameters remain local.
    """
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_personalized_indices(strategy_name: str) -> list[int]:
    """
    Returns the indices of the parameters in get_parameters() that should 
    remain strictly local for the given personalized FL algorithm.
    """
    if strategy_name == "fedper":
        # FedPerNet state_dict:
        # base.0.weight, base.0.bias (0, 1)
        # base.3.weight, base.3.bias (2, 3)
        # head.weight, head.bias (4, 5)
        return [4, 5]
    elif strategy_name == "fedbn":
        # FedBNNet state_dict:
        # network.0.weight, bias (0, 1)
        # network.1.weight, bias, running_mean, running_var, num_batches_tracked (2, 3, 4, 5, 6)
        # network.4.weight, bias (7, 8)
        # network.5.weight, bias, running_mean, running_var, num_batches_tracked (9, 10, 11, 12, 13)
        # network.8.weight, bias (14, 15)
        # Local BN layers: indices 2 to 6, and 9 to 13.
        return [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    return []
