"""
PHASE 9 - DYNAMIC CREDENTIAL RISK SCORE (DCRS) ENGINEERING
Aggregates contextual login features -- login_attempts,
browser_tab_count, typing_speed_wpm, risk_label, bdi_score, and
abfe_score -- into a single continuous risk indicator per login
attempt, completing the three-metric behavioural feature-engineering
stage (BDI + ABFE + DCRS).
"""
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DCRS_FEATURES = [
    "login_attempts",
    "browser_tab_count",
    "typing_speed_wpm",
    "risk_label",
    "bdi_score",
    "abfe_score",
]


def prepare_dcrs_features(df):
    present = [c for c in DCRS_FEATURES if c in df.columns]
    missing = [c for c in DCRS_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] DCRS features missing, filled with 0: {missing}")
        for c in missing:
            df[c] = 0
    print("Step 36 - Selected DCRS features:", DCRS_FEATURES)

    work = df[DCRS_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    work = pd.DataFrame(MinMaxScaler().fit_transform(work), columns=DCRS_FEATURES, index=df.index)
    print("Step 37 - Prepared and scaled DCRS features (MinMaxScaler).")
    return work


def compute_dcrs(df):
    work = prepare_dcrs_features(df)
    df["dcrs_score"] = work.mean(axis=1)
    print("\nStep 38 - Computed Dynamic Credential Risk Score (DCRS):")
    print(df[["dcrs_score"]].describe())
    return df


if __name__ == "__main__":
    print("=== PHASE 9: DYNAMIC CREDENTIAL RISK SCORE (DCRS) ENGINEERING ===\n")
    in_path = os.path.join(OUT_DIR, "dataset_with_abfe.csv")
    df = pd.read_csv(in_path)
    df = compute_dcrs(df)

    print("\nStep 39 - Verification (sample DCRS scores):")
    print(df[["dcrs_score"]].head(10))

    out_path = os.path.join(OUT_DIR, "dataset_with_dcrs.csv")
    df.to_csv(out_path, index=False)
    print(f"\nStep 40-42 - Saved final feature-engineered dataset -> {out_path}")
    print("Phase 9 completion: BDI + ABFE + DCRS all present in final dataset.")
