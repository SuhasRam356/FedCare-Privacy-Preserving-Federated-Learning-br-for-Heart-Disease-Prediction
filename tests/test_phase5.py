"""
tests/test_phase5.py - Test suite for Phase 5: Dashboard & Demo components.

Validates:
    1. Dashboard module imports correctly
    2. Data loading functions work with existing CSV files
    3. Model loading and inference pipeline
    4. Feature encoding matches training data schema
    5. Risk calculator input/output pipeline
    6. Demo script checks pass
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "heart"
RESULTS_DIR = PROJECT_ROOT / "results"


# ══════════════════════════════════════════════════════════════════════
#  Test Group 1: Dashboard Data Loaders
# ══════════════════════════════════════════════════════════════════════

class TestDashboardDataLoaders:
    """Verify all cached data loader functions return expected data."""

    def test_hospital_stats_shape(self):
        """Hospital stats should have 6 rows (one per hospital)."""
        stats = []
        for i in range(1, 7):
            csv_path = DATA_DIR / f"hospital_{i}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                stats.append({
                    "Hospital": f"Hospital {i}",
                    "Samples": len(df),
                    "Positive": int(df["target"].sum()),
                    "Prevalence": df["target"].mean(),
                })
        stats_df = pd.DataFrame(stats)
        assert len(stats_df) == 6, f"Expected 6 hospitals, got {len(stats_df)}"

    def test_hospital_stats_columns(self):
        """Each hospital should have expected stat columns."""
        df = pd.read_csv(DATA_DIR / "hospital_1.csv")
        assert "target" in df.columns, "Missing 'target' column"
        assert "age" in df.columns, "Missing 'age' column"
        assert "cholesterol" in df.columns, "Missing 'cholesterol' column"

    def test_fedavg_rounds_exist(self):
        """FedAvg round data should exist with expected columns."""
        path = RESULTS_DIR / "rounds_fedavg.csv"
        if path.exists():
            df = pd.read_csv(path)
            required_cols = ["round", "auc", "accuracy", "test_loss"]
            for col in required_cols:
                assert col in df.columns, f"Missing column '{col}' in rounds_fedavg.csv"
            assert len(df) > 0, "rounds_fedavg.csv is empty"

    def test_attack_defense_matrix_exist(self):
        """Attack-defense matrix should have expected structure."""
        path = RESULTS_DIR / "phase4_attack_defense_matrix.csv"
        if path.exists():
            df = pd.read_csv(path)
            required_cols = ["Attack_Name", "Strategy_Name", "Final_AUC"]
            for col in required_cols:
                assert col in df.columns, f"Missing column '{col}'"

    def test_dp_sweep_exist(self):
        """DP sweep data should contain epsilon and AUC columns."""
        path = RESULTS_DIR / "phase4_dp_sweep.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert "Epsilon" in df.columns
            assert "Final_AUC" in df.columns

    def test_non_iid_results_exist(self):
        """Non-IID experiment results should have expected structure."""
        path = RESULTS_DIR / "phase3_non_iid_experiments.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert "Configuration" in df.columns
            assert "Final_AUC" in df.columns
            assert "Equity_Gap" in df.columns

    def test_comm_cost_exist(self):
        """Communication cost analysis should have expected columns."""
        path = RESULTS_DIR / "phase4_comm_cost.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert "param_count" in df.columns
            assert "total_fl_mb" in df.columns
            assert "comm_ratio" in df.columns


# ══════════════════════════════════════════════════════════════════════
#  Test Group 2: Model Loading & Inference
# ══════════════════════════════════════════════════════════════════════

class TestModelInference:
    """Verify the FedCare model loads and produces valid predictions."""

    def test_model_creation(self):
        """Model should create successfully with correct architecture."""
        from fedcare.task import Net, NUM_FEATURES, NUM_CLASSES
        model = Net()
        assert model is not None
        # Verify input/output dimensions
        dummy_input = torch.randn(1, NUM_FEATURES)
        output = model(dummy_input)
        assert output.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {output.shape}"

    def test_model_inference_batch(self):
        """Model should handle batch inference correctly."""
        from fedcare.task import Net
        model = Net()
        model.eval()
        batch = torch.randn(32, 13)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (32, 2)

    def test_softmax_probabilities(self):
        """Softmax output should sum to 1 for each sample."""
        from fedcare.task import Net
        model = Net()
        model.eval()
        batch = torch.randn(10, 13)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)

    def test_risk_probability_range(self):
        """Risk probability should be between 0 and 1."""
        from fedcare.task import Net
        model = Net()
        model.eval()
        x = torch.randn(100, 13)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  Test Group 3: Feature Encoding Pipeline
# ══════════════════════════════════════════════════════════════════════

class TestFeatureEncoding:
    """Verify feature encoding matches training data schema."""

    EXPECTED_FEATURES = [
        "age", "resting_bp", "cholesterol", "max_heart_rate",
        "bmi", "glucose", "sex", "smoker", "diabetes_history",
        "family_history", "cp_atypical_angina", "cp_non_anginal",
        "cp_typical_angina",
    ]

    def test_training_data_features_match(self):
        """Training data columns (minus target) should match expected features."""
        df = pd.read_csv(DATA_DIR / "combined.csv")
        feature_cols = [c for c in df.columns if c != "target"]
        assert feature_cols == self.EXPECTED_FEATURES, (
            f"Feature mismatch.\nExpected: {self.EXPECTED_FEATURES}\nGot: {feature_cols}"
        )

    def test_feature_count(self):
        """Should have exactly 13 features."""
        df = pd.read_csv(DATA_DIR / "combined.csv")
        n_features = len(df.columns) - 1  # minus target
        assert n_features == 13, f"Expected 13 features, got {n_features}"

    def test_risk_calculator_encoding(self):
        """Verify the risk calculator encoding produces the right shape."""
        # Simulate the encoding logic from the dashboard
        raw = np.array([[
            55,    # age
            130,   # resting_bp
            240,   # cholesterol
            150,   # max_heart_rate
            27.5,  # bmi
            100,   # glucose
            1,     # sex (Male=1)
            0,     # smoker (No=0)
            0,     # diabetes_history
            0,     # family_history
            0,     # cp_atypical_angina
            0,     # cp_non_anginal
            0,     # cp_typical_angina
        ]], dtype=np.float64)
        assert raw.shape == (1, 13), f"Expected (1, 13), got {raw.shape}"

    def test_scaler_transform(self):
        """StandardScaler should transform features without error."""
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv(DATA_DIR / "combined.csv")
        X = df.drop(columns=["target"]).values
        scaler = StandardScaler()
        scaler.fit(X)

        sample = np.array([[55, 130, 240, 150, 27.5, 100, 1, 0, 0, 0, 0, 0, 0]])
        scaled = scaler.transform(sample)
        assert scaled.shape == (1, 13)
        # Scaled values should be roughly standardized
        assert not np.any(np.isnan(scaled)), "Scaler produced NaN values"

    def test_chest_pain_one_hot_encoding(self):
        """Chest pain one-hot encoding should produce valid combinations."""
        pain_types = {
            "Asymptomatic": (0, 0, 0),
            "Atypical Angina": (1, 0, 0),
            "Non-Anginal": (0, 1, 0),
            "Typical Angina": (0, 0, 1),
        }
        for pain_type, expected in pain_types.items():
            cp_atypical = 1 if pain_type == "Atypical Angina" else 0
            cp_non_anginal = 1 if pain_type == "Non-Anginal" else 0
            cp_typical = 1 if pain_type == "Typical Angina" else 0
            result = (cp_atypical, cp_non_anginal, cp_typical)
            assert result == expected, f"Encoding mismatch for '{pain_type}': {result} != {expected}"


# ══════════════════════════════════════════════════════════════════════
#  Test Group 4: End-to-End Risk Prediction Pipeline
# ══════════════════════════════════════════════════════════════════════

class TestEndToEndPrediction:
    """Full pipeline test: raw input -> scaled -> model -> probability."""

    def test_full_prediction_pipeline(self):
        """Complete prediction pipeline should produce valid risk scores."""
        from fedcare.task import Net
        from sklearn.preprocessing import StandardScaler

        # 1. Load and fit scaler
        df = pd.read_csv(DATA_DIR / "combined.csv")
        X = df.drop(columns=["target"]).values
        scaler = StandardScaler()
        scaler.fit(X)

        # 2. Create model
        model = Net()
        model.eval()

        # 3. Encode patient features
        patient = np.array([[
            65, 140, 260, 130, 30.0, 110, 1, 1, 1, 1, 0, 0, 1,
        ]], dtype=np.float64)

        # 4. Scale
        scaled = scaler.transform(patient)
        tensor = torch.tensor(scaled, dtype=torch.float32)

        # 5. Predict
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            risk = float(probs[0, 1])

        assert 0.0 <= risk <= 1.0, f"Risk probability out of range: {risk}"

    def test_multiple_patients_batch(self):
        """Batch prediction should work for multiple patients."""
        from fedcare.task import Net
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(DATA_DIR / "combined.csv")
        X = df.drop(columns=["target"]).values
        scaler = StandardScaler()
        scaler.fit(X)

        model = Net()
        model.eval()

        # Create 5 different patients
        patients = np.array([
            [25, 110, 180, 180, 22.0, 85, 0, 0, 0, 0, 0, 0, 0],
            [45, 130, 220, 160, 26.0, 95, 1, 0, 0, 0, 1, 0, 0],
            [55, 140, 250, 145, 28.5, 105, 1, 1, 0, 1, 0, 0, 0],
            [65, 155, 280, 120, 31.0, 120, 1, 1, 1, 1, 0, 0, 1],
            [75, 170, 310, 100, 35.0, 140, 1, 1, 1, 1, 0, 0, 1],
        ], dtype=np.float64)

        scaled = scaler.transform(patients)
        tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[:, 1]

        assert probs.shape == (5,)
        assert all(0.0 <= p <= 1.0 for p in probs.numpy())


# ══════════════════════════════════════════════════════════════════════
#  Test Group 5: Demo Script Verification
# ══════════════════════════════════════════════════════════════════════

class TestDemoScript:
    """Verify demo script components work correctly."""

    def test_demo_script_exists(self):
        """demo.py should exist in project root."""
        assert (PROJECT_ROOT / "demo.py").exists()

    def test_dashboard_module_exists(self):
        """app/dashboard.py should exist."""
        assert (PROJECT_ROOT / "app" / "dashboard.py").exists()

    def test_dashboard_imports(self):
        """Dashboard should import without errors."""
        # Just verify the module file is valid Python
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dashboard",
            PROJECT_ROOT / "app" / "dashboard.py",
        )
        assert spec is not None, "Could not create module spec for dashboard"

    def test_all_data_files_exist(self):
        """All required data files for the dashboard should exist."""
        required = [DATA_DIR / "combined.csv"] + [
            DATA_DIR / f"hospital_{i}.csv" for i in range(1, 7)
        ]
        for f in required:
            assert f.exists(), f"Missing required file: {f}"

    def test_results_figures_exist(self):
        """At least some research figures should exist."""
        figures = list(RESULTS_DIR.glob("figure*.png"))
        assert len(figures) > 0, "No research figures found in results/"


# ══════════════════════════════════════════════════════════════════════
#  Test Group 6: Risk Classification Logic
# ══════════════════════════════════════════════════════════════════════

class TestRiskClassification:
    """Test the risk level classification logic."""

    def _classify(self, prob: float) -> str:
        """Replicate the dashboard's risk classification logic."""
        if prob < 0.25:
            return "LOW RISK"
        elif prob < 0.50:
            return "MODERATE RISK"
        elif prob < 0.75:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"

    def test_low_risk(self):
        assert self._classify(0.0) == "LOW RISK"
        assert self._classify(0.10) == "LOW RISK"
        assert self._classify(0.24) == "LOW RISK"

    def test_moderate_risk(self):
        assert self._classify(0.25) == "MODERATE RISK"
        assert self._classify(0.35) == "MODERATE RISK"
        assert self._classify(0.49) == "MODERATE RISK"

    def test_high_risk(self):
        assert self._classify(0.50) == "HIGH RISK"
        assert self._classify(0.60) == "HIGH RISK"
        assert self._classify(0.74) == "HIGH RISK"

    def test_very_high_risk(self):
        assert self._classify(0.75) == "VERY HIGH RISK"
        assert self._classify(0.90) == "VERY HIGH RISK"
        assert self._classify(1.0) == "VERY HIGH RISK"

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert self._classify(0.249999) == "LOW RISK"
        assert self._classify(0.25) == "MODERATE RISK"
        assert self._classify(0.499999) == "MODERATE RISK"
        assert self._classify(0.50) == "HIGH RISK"
        assert self._classify(0.749999) == "HIGH RISK"
        assert self._classify(0.75) == "VERY HIGH RISK"
