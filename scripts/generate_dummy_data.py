"""
Generate synthetic heart-disease data for 6 hospitals.

Each hospital gets a slightly different data distribution (varying sample
sizes and class imbalance) to simulate realistic Non-IID skew that will
be explored further in Phase 2.

Features (12 continuous):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca

Target:
    target  (0 = no disease, 1 = disease)

Outputs:
    data/heart/hospital_1.csv  …  data/heart/hospital_6.csv
    data/heart/combined.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# ── Feature definitions ──────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "resting_bp", "cholesterol", "max_heart_rate", "bmi", "glucose",
    "sex", "smoker", "diabetes_history", "family_history",
    "cp_atypical_angina", "cp_non_anginal", "cp_typical_angina",
]

# Realistic ranges (loosely based on the Cleveland dataset)
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":                (29.0, 77.0),
    "resting_bp":         (94.0, 200.0),
    "cholesterol":        (126.0, 564.0),
    "max_heart_rate":     (71.0, 202.0),
    "bmi":                (18.0, 40.0),
    "glucose":            (70.0, 200.0),
    "sex":                (0.0,  1.0),
    "smoker":             (0.0,  1.0),
    "diabetes_history":   (0.0,  1.0),
    "family_history":     (0.0,  1.0),
    "cp_atypical_angina": (0.0,  1.0),
    "cp_non_anginal":     (0.0,  1.0),
    "cp_typical_angina":  (0.0,  1.0),
}

# Per-hospital configuration: (num_samples, positive_class_ratio)
# Varying sizes and ratios simulates realistic Non-IID skew.
HOSPITAL_CONFIG: list[tuple[int, float]] = [
    (2000, 0.235),   # Hospital 1
    (2000, 0.048),   # Hospital 2
    (2000, 0.379),   # Hospital 3
    (2000, 0.207),   # Hospital 4
    (2000, 0.463),   # Hospital 5
    (2000, 0.342),   # Hospital 6
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "heart"


def _generate_hospital(n_samples: int, pos_ratio: float) -> pd.DataFrame:
    """Return a DataFrame with synthetic heart-disease records."""
    data: dict[str, np.ndarray] = {}

    for feat in FEATURE_NAMES:
        lo, hi = FEATURE_RANGES[feat]
        if feat in ("sex", "fbs", "exang"):
            # Binary features
            data[feat] = rng.integers(int(lo), int(hi) + 1, size=n_samples).astype(float)
        elif feat in ("cp", "restecg", "slope", "ca"):
            # Discrete ordinal features
            data[feat] = rng.integers(int(lo), int(hi) + 1, size=n_samples).astype(float)
        else:
            # Continuous features
            data[feat] = rng.uniform(lo, hi, size=n_samples)

    # Target: controlled positive ratio
    n_pos = int(n_samples * pos_ratio)
    target = np.array([1] * n_pos + [0] * (n_samples - n_pos))
    rng.shuffle(target)
    data["target"] = target.astype(float)

    return pd.DataFrame(data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for idx, (n_samples, pos_ratio) in enumerate(HOSPITAL_CONFIG, start=1):
        df = _generate_hospital(n_samples, pos_ratio)
        path = OUTPUT_DIR / f"hospital_{idx}.csv"
        df.to_csv(path, index=False)
        print(f"[OK] hospital_{idx}.csv  --  {len(df)} rows, "
              f"pos_ratio={df['target'].mean():.2f}  ->  {path}")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n[OK] combined.csv  --  {len(combined)} rows  ->  {combined_path}")


if __name__ == "__main__":
    main()
