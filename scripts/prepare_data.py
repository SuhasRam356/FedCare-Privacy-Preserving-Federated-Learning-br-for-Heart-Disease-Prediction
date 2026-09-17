"""
scripts/prepare_data.py -- Extract and preprocess the real hospital data.

Reads the multi-sheet Excel file, encodes categorical features, and writes
per-hospital CSVs + a combined CSV to data/heart/.

Encoding strategy:
    - sex:              F=0, M=1  (binary)
    - chest_pain_type:  one-hot (4 columns, drop_first=True -> 3 dummies)
    - smoker:           no=0, yes=1
    - diabetes_history: no=0, yes=1
    - family_history:   no=0, yes=1
    - disease_label:    already 0/1 -> renamed to 'target'

Result: 6 continuous + 1 sex + 3 chest_pain dummies + 3 binary = 13 features + target

Usage:
    python scripts/prepare_data.py
"""

from pathlib import Path
import pandas as pd

EXCEL_PATH = Path(__file__).resolve().parent.parent / "data" / "heart" / "Heart_Disease_Multi-Hospital_Research_Database.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "heart"
HOSPITAL_SHEETS = ["Hospital 1", "Hospital 2", "Hospital 3", "Hospital 4", "Hospital 5", "Hospital 6"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and rename target column."""
    # Drop hospital_id (not a feature)
    df = df.drop(columns=["hospital_id"], errors="ignore")

    # Binary encode
    df["sex"] = (df["sex"] == "M").astype(int)
    df["smoker"] = (df["smoker"] == "yes").astype(int)
    df["diabetes_history"] = (df["diabetes_history"] == "yes").astype(int)
    df["family_history"] = (df["family_history"] == "yes").astype(int)

    # One-hot encode chest_pain_type (drop first to avoid multicollinearity)
    chest_dummies = pd.get_dummies(df["chest_pain_type"], prefix="cp", drop_first=True, dtype=int)
    df = df.drop(columns=["chest_pain_type"])
    df = pd.concat([df, chest_dummies], axis=1)

    # Rename target
    df = df.rename(columns={"disease_label": "target"})

    # Ensure target is last column
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    return df


def main() -> None:
    print(f"Reading: {EXCEL_PATH}")
    xls = pd.ExcelFile(EXCEL_PATH)

    all_frames: list[pd.DataFrame] = []

    for idx, sheet in enumerate(HOSPITAL_SHEETS, start=1):
        df_raw = pd.read_excel(xls, sheet)
        df = preprocess(df_raw)
        path = OUTPUT_DIR / f"hospital_{idx}.csv"
        df.to_csv(path, index=False)

        n_pos = df["target"].sum()
        pos_ratio = n_pos / len(df)
        print(f"[OK] hospital_{idx}.csv -- {len(df)} rows, "
              f"pos_ratio={pos_ratio:.3f} ({int(n_pos)} positive) -> {path}")
        all_frames.append(df)

    # Combined
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n[OK] combined.csv -- {len(combined)} rows -> {combined_path}")

    # Print feature summary
    feature_cols = [c for c in combined.columns if c != "target"]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution:\n{combined['target'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
