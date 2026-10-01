"""
PHASE 8 - ADAPTIVE BEHAVIOURAL FINGERPRINT ENTROPY (ABFE) ENGINEERING
Fuses multiple weighted behavioural indicators into a single
entropy-based unpredictability score per login attempt, using the
features documented in the project report: password_length,
uses_special_char, login_attempts, browser_tab_count,
typing_speed_wpm, and risk_label.
"""
import os
import pandas as pd

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ABFE_FEATURES = [
    "password_length",
    "uses_special_char",
    "login_attempts",
    "browser_tab_count",
    "typing_speed_wpm",
    "risk_label",
]

# Weights reflect each indicator's relative contribution to behavioural
# unpredictability; equal weighting is used as the default baseline.
DEFAULT_WEIGHTS = {c: 1.0 / len(ABFE_FEATURES) for c in ABFE_FEATURES}


def prepare_abfe_features(df):
    present = [c for c in ABFE_FEATURES if c in df.columns]
    missing = [c for c in ABFE_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] ABFE features missing, filled with 0: {missing}")
        for c in missing:
            df[c] = 0
    print("Step 28 - Selected ABFE features:", ABFE_FEATURES)

    work = df[ABFE_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in ABFE_FEATURES:
        col_max = work[col].max()
        work[col] = work[col] / col_max if col_max != 0 else 0.0
    print("Step 29 - Prepared and normalized ABFE features.")
    return work


def compute_abfe(df):
    work = prepare_abfe_features(df)
    weighted = sum(work[c] * DEFAULT_WEIGHTS[c] for c in ABFE_FEATURES)
    df["abfe_score"] = weighted
    print("\nStep 30 - Computed Adaptive Behavioural Fingerprint Entropy (ABFE):")
    print(df[["abfe_score"]].describe())
    return df


if __name__ == "__main__":
    print("=== PHASE 8: ADAPTIVE BEHAVIOURAL FINGERPRINT ENTROPY (ABFE) ENGINEERING ===\n")
    in_path = os.path.join(OUT_DIR, "dataset_with_bdi.csv")
    df = pd.read_csv(in_path)
    df = compute_abfe(df)

    print("\nStep 31 - Verification (sample ABFE scores):")
    print(df[["abfe_score"]].head(10))

    out_path = os.path.join(OUT_DIR, "dataset_with_abfe.csv")
    df.to_csv(out_path, index=False)
    print(f"\nStep 32-34 - Saved dataset with ABFE -> {out_path}")
