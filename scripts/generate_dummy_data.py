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
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
]

# Realistic ranges (loosely based on the Cleveland dataset)
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":      (29.0, 77.0),
    "sex":      (0.0,  1.0),    # binary
    "cp":       (0.0,  3.0),    # chest-pain type
    "trestbps": (94.0, 200.0),
    "chol":     (126.0, 564.0),
    "fbs":      (0.0,  1.0),    # binary
    "restecg":  (0.0,  2.0),
    "thalach":  (71.0, 202.0),
    "exang":    (0.0,  1.0),    # binary
    "oldpeak":  (0.0,  6.2),
    "slope":    (0.0,  2.0),
    "ca":       (0.0,  4.0),
}

# Per-hospital configuration: (num_samples, positive_class_ratio)
# Varying sizes and ratios simulates mild Non-IID skew.
HOSPITAL_CONFIG: list[tuple[int, float]] = [
    (300, 0.45),   # Hospital 1 – large, balanced
    (250, 0.50),   # Hospital 2 – medium, balanced
    (180, 0.35),   # Hospital 3 – smaller, fewer positives
    (220, 0.55),   # Hospital 4 – medium, more positives
    (150, 0.40),   # Hospital 5 – small
    (200, 0.60),   # Hospital 6 – skewed toward disease
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
