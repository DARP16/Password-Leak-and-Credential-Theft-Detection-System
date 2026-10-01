"""
PHASE 6 - DATA CLEANING AND UNIFICATION
Cleans each raw dataset (missing values, duplicates, column-name
standardization), compares column sets across sources to identify
common vs. unique fields, then merges them into a single unified
base dataset carrying the target risk_label column used for all
downstream feature engineering (Phases 7-9) and model training
(Phase 10).
"""
import os
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = {
    "auth": "auth_authz_failures_dataset.csv",
    "password": "password_security_dataset.csv",
    "rba": "rba_cleaned.csv",
}


def standardize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean_dataset(df, name):
    before = df.shape[0]
    df = df.drop_duplicates()
    dup_removed = before - df.shape[0]

    missing_cols = df.columns[df.isna().any()].tolist()
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    print(f"[{name}] duplicates removed: {dup_removed} | missing-value columns handled: {missing_cols}")
    return df


def load_and_clean_all():
    frames = {}
    for key, filename in SOURCES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[WARN] {filename} not found, skipping {key}")
            continue
        df = pd.read_csv(path)
        df = standardize_columns(df)
        df = clean_dataset(df, key)
        frames[key] = df
    return frames


def compare_columns(frames):
    print("\n=== Step 11-12: Compare Dataset Columns ===")
    col_sets = {k: set(v.columns) for k, v in frames.items()}
    all_cols = set().union(*col_sets.values())
    common_cols = set.intersection(*col_sets.values()) if len(col_sets) > 1 else all_cols
    print("Common columns across all sources:", sorted(common_cols))
    for k, cols in col_sets.items():
        unique = cols - common_cols
        print(f"Unique to {k}: {sorted(unique)}")
    return common_cols


def build_unified_dataset(frames, common_cols):
    print("\n=== Step 16-17: Identify Common Fields & Build Unified Base Dataset ===")
    # Target-relevant behavioural columns kept even if not common to every source,
    # since they are what Phases 7-9 depend on.
    keep_cols = sorted(common_cols | {
        "login_attempts", "typing_speed_wpm", "browser_tab_count",
        "challenge_sequence", "timezone_offset_hours", "session_id",
        "password_length", "uses_special_char", "capslock_on_flag",
        "short_password_flag", "many_login_attempts_flag", "fast_typing_flag",
        "risk_score_heuristic", "risk_label",
    })

    unified_frames = []
    for name, df in frames.items():
        present = [c for c in keep_cols if c in df.columns]
        sub = df[present].copy()
        sub["source"] = name
        unified_frames.append(sub)

    unified = pd.concat(unified_frames, ignore_index=True, sort=False)

    if "risk_label" not in unified.columns:
        raise ValueError("risk_label column missing from all sources — cannot build target variable.")

    print("Unified base dataset shape:", unified.shape)
    print("Target variable distribution:\n", unified["risk_label"].value_counts())
    return unified


if __name__ == "__main__":
    print("=== PHASE 6: DATA CLEANING AND UNIFICATION ===\n")
    frames = load_and_clean_all()
    if not frames:
        raise SystemExit("No source datasets found in data/. Place the Phase 5 files there first.")
    common_cols = compare_columns(frames)
    unified = build_unified_dataset(frames, common_cols)
    out_path = os.path.join(OUT_DIR, "unified_base_dataset.csv")
    unified.to_csv(out_path, index=False)
    print(f"\nSaved unified base dataset -> {out_path}")
