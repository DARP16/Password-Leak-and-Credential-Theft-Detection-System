"""
PHASE 5 - DATASET IMPORT
Mounts the staged data folder (Google Drive in the original Colab run,
a local `data/` folder here) and verifies that all raw datasets produced
by Phase 4 (log collection) and the supporting reference datasets are
present, readable, and structurally sound before any cleaning begins.
"""
import os
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

EXPECTED_FILES = [
    "auth_authz_failures_dataset.csv",   # merged auth.log + Zeek + Suricata + Auditd failure events
    "password_security_dataset.csv",
    "password_security_dataset_features.csv",
    "password_security_train.csv",
    "password_security_test.csv",
    "rba_cleaned.csv",                   # cleaned Risk-Based Authentication reference dataset
]


def verify_datasets():
    print("=== PHASE 5: DATASET IMPORT & VERIFICATION ===\n")
    found = {}
    for name in EXPECTED_FILES:
        path = os.path.join(DATA_DIR, name)
        exists = os.path.exists(path)
        print(f"{name:45s} {'FOUND' if exists else 'NOT FOUND'}")
        if exists:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_excel(path)
            print(f"   shape={df.shape}  columns={list(df.columns)[:8]}{'...' if df.shape[1] > 8 else ''}")
            found[name] = df.shape
        else:
            found[name] = None
    print("\nImported datasets:")
    for name, shape in found.items():
        print(f" - {name}: {'OK ' + str(shape) if shape else 'MISSING'}")
    return found


if __name__ == "__main__":
    verify_datasets()
