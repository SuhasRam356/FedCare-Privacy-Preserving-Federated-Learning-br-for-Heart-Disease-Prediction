"""
fedcare/federated_xgboost.py – Federated Tree-Based Models for FedCare.

Implements Federated XGBoost and Federated Random Forest via a
histogram-sharing simulation approach:
    - Each hospital trains a local XGBoost/RF model
    - Ensemble predictions are aggregated (soft voting)
    - Performance is compared against the federated MLP

This is a simulation-level implementation matching FedCare's sequential
in-process design pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "heart"
RANDOM_STATE = 42
TEST_SPLIT = 0.2


def _load_hospital_data(hospital_id: int):
    """Load and split a single hospital's data."""
    csv_path = DATA_DIR / f"hospital_{hospital_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _get_global_test_set():
    """Assemble global test set from all hospitals (same split as MLP)."""
    X_te_list, y_te_list = [], []
    for i in range(1, 7):
        _, X_te, _, y_te = _load_hospital_data(i)
        X_te_list.append(X_te)
        y_te_list.append(y_te)
    return np.vstack(X_te_list), np.concatenate(y_te_list)


class FederatedXGBoost:
    """
    Federated XGBoost using ensemble aggregation.

    Each hospital trains a local XGBoost classifier.
    Global predictions are the weighted average of local prediction
    probabilities (soft voting / federated ensemble).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_hospitals: int = 6,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_hospitals = n_hospitals
        self.local_models = []
        self.local_scalers = []
        self.local_sample_counts = []

    def train(self) -> dict:
        """Train local XGBoost models at each hospital."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install: pip install xgboost")

        self.local_models = []
        self.local_scalers = []
        self.local_sample_counts = []
        local_results = []

        for i in range(1, self.n_hospitals + 1):
            X_train, X_test, y_train, y_test = _load_hospital_data(i)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
            model.fit(X_train_s, y_train)

            y_pred_proba = model.predict_proba(X_test_s)[:, 1]
            y_pred = model.predict(X_test_s)
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)

            self.local_models.append(model)
            self.local_scalers.append(scaler)
            self.local_sample_counts.append(len(X_train))

            local_results.append({
                "Hospital": f"Hospital {i}",
                "AUC": round(auc, 4),
                "Accuracy": round(acc, 4),
                "Samples": len(X_train),
            })

        return {"local_results": local_results}

    def predict_global(self, X_raw: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction across all hospital models."""
        total_samples = sum(self.local_sample_counts)
        weighted_probs = np.zeros(len(X_raw))

        for model, scaler, n in zip(
            self.local_models, self.local_scalers, self.local_sample_counts
        ):
            X_scaled = scaler.transform(X_raw)
            probs = model.predict_proba(X_scaled)[:, 1]
            weighted_probs += probs * (n / total_samples)

        return weighted_probs

    def evaluate_global(self) -> dict:
        """Evaluate ensemble performance on the global test set."""
        X_test, y_test = _get_global_test_set()
        probs = self.predict_global(X_test)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)

        return {"global_auc": round(auc, 4), "global_accuracy": round(acc, 4)}


class FederatedRandomForest:
    """
    Federated Random Forest using ensemble aggregation.

    Each hospital trains a local Random Forest.
    Global prediction is the weighted soft-vote ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 12,
        n_hospitals: int = 6,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_hospitals = n_hospitals
        self.local_models = []
        self.local_scalers = []
        self.local_sample_counts = []

    def train(self) -> dict:
        """Train local RF models at each hospital."""
        self.local_models = []
        self.local_scalers = []
        self.local_sample_counts = []
        local_results = []

        for i in range(1, self.n_hospitals + 1):
            X_train, X_test, y_train, y_test = _load_hospital_data(i)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train_s, y_train)

            y_pred_proba = model.predict_proba(X_test_s)[:, 1]
            y_pred = model.predict(X_test_s)
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)

            self.local_models.append(model)
            self.local_scalers.append(scaler)
            self.local_sample_counts.append(len(X_train))

            local_results.append({
                "Hospital": f"Hospital {i}",
                "AUC": round(auc, 4),
                "Accuracy": round(acc, 4),
                "Samples": len(X_train),
            })

        return {"local_results": local_results}

    def predict_global(self, X_raw: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        total_samples = sum(self.local_sample_counts)
        weighted_probs = np.zeros(len(X_raw))

        for model, scaler, n in zip(
            self.local_models, self.local_scalers, self.local_sample_counts
        ):
            X_scaled = scaler.transform(X_raw)
            probs = model.predict_proba(X_scaled)[:, 1]
            weighted_probs += probs * (n / total_samples)

        return weighted_probs

    def evaluate_global(self) -> dict:
        """Evaluate ensemble performance on the global test set."""
        X_test, y_test = _get_global_test_set()
        probs = self.predict_global(X_test)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)

        return {"global_auc": round(auc, 4), "global_accuracy": round(acc, 4)}


def run_model_comparison() -> pd.DataFrame:
    """
    Run all three federated model types and return a comparison table.

    Returns a DataFrame with columns: Model, Global_AUC, Global_Accuracy
    """
    from fedcare.task import Net, load_data, evaluate

    # 1. Federated MLP (existing)
    mlp = Net()
    train_loader, test_loader, _ = load_data(partition_id=None)
    from fedcare.task import train as train_fn
    train_fn(mlp, train_loader, epochs=15, lr=0.001)
    mlp_metrics = evaluate(mlp, test_loader)

    results = [
        {"Model": "Federated MLP (FedAvg)", "Global_AUC": round(mlp_metrics["auc"], 4),
         "Global_Accuracy": round(mlp_metrics["accuracy"], 4)},
    ]

    # 2. Federated XGBoost
    try:
        fed_xgb = FederatedXGBoost()
        fed_xgb.train()
        xgb_metrics = fed_xgb.evaluate_global()
        results.append({
            "Model": "Federated XGBoost",
            "Global_AUC": xgb_metrics["global_auc"],
            "Global_Accuracy": xgb_metrics["global_accuracy"],
        })
    except ImportError:
        results.append({"Model": "Federated XGBoost", "Global_AUC": 0.0, "Global_Accuracy": 0.0})

    # 3. Federated Random Forest
    fed_rf = FederatedRandomForest()
    fed_rf.train()
    rf_metrics = fed_rf.evaluate_global()
    results.append({
        "Model": "Federated Random Forest",
        "Global_AUC": rf_metrics["global_auc"],
        "Global_Accuracy": rf_metrics["global_accuracy"],
    })

    return pd.DataFrame(results)
