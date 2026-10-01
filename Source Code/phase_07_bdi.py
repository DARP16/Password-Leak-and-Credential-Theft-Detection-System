"""
PHASE 7 - BEHAVIOUR DRIFT INDEX (BDI) ENGINEERING
Computes the Behaviour Drift Index for every login attempt using the
six behavioural features documented in the project report:
login_attempts, typing_speed_wpm, browser_tab_count, challenge_sequence,
timezone_offset_hours, and session_id. Numerical features are
normalized (min-max), categorical/ordinal features are label-encoded,
and the BDI score is the mean of the normalized feature vector.
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BDI_FEATURES = [
    "login_attempts",
    "typing_speed_wpm",
    "browser_tab_count",
    "challenge_sequence",
    "timezone_offset_hours",
    "session_id",
]


def select_bdi_features(df):
    present = [c for c in BDI_FEATURES if c in df.columns]
    missing = [c for c in BDI_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] BDI features missing from dataset, filled with 0: {missing}")
        for c in missing:
            df[c] = 0
    print("Step 19 - Selected BDI features:", BDI_FEATURES)
    return df, BDI_FEATURES


def normalize_and_encode(df, features):
    work = df[features].copy()
    for col in features:
        if pd.api.types.is_numeric_dtype(work[col]):
            col_min, col_max = work[col].min(), work[col].max()
            work[col] = (work[col] - col_min) / (col_max - col_min) if col_max != col_min else 0.0
        else:
            work[col] = LabelEncoder().fit_transform(work[col].astype(str))
            col_max = work[col].max()
            work[col] = work[col] / col_max if col_max != 0 else 0.0
    print("Step 20-21 - Normalized numerical features and encoded categorical features.")
    return work


def compute_bdi(df):
    df, features = select_bdi_features(df)
    normalized = normalize_and_encode(df, features)
    df["bdi_score"] = normalized.mean(axis=1)
    print("\nStep 22 - Computed Behaviour Drift Index (BDI):")
    print(df[["bdi_score"]].describe())
    return df


if __name__ == "__main__":
    print("=== PHASE 7: BEHAVIOUR DRIFT INDEX (BDI) ENGINEERING ===\n")
    in_path = os.path.join(OUT_DIR, "unified_base_dataset.csv")
    df = pd.read_csv(in_path)
    df = compute_bdi(df)

    print("\nStep 23 - Verification (sample BDI scores):")
    print(df[["bdi_score"]].head(10))

    out_path = os.path.join(OUT_DIR, "dataset_with_bdi.csv")
    df.to_csv(out_path, index=False)
    print(f"\nStep 24-26 - Saved dataset with BDI -> {out_path}")
