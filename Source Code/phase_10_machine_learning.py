"""
PHASE 10 - MACHINE LEARNING MODEL TRAINING AND COMPARISON
Trains and compares FIVE supervised classifiers (Random Forest,
XGBoost, Logistic Regression, Decision Tree, Support Vector Machine)
on the final feature-engineered dataset, then retrains Random Forest
without the risk_score_heuristic feature as a feature-leakage check,
before saving the best-performing, leakage-checked model.
"""
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — run: pip install xgboost")

OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "password_length", "login_attempts", "browser_tab_count", "typing_speed_wpm",
    "uses_special_char", "capslock_on_flag", "short_password_flag",
    "many_login_attempts_flag", "fast_typing_flag", "timezone_offset_hours",
    "risk_score_heuristic", "bdi_score", "abfe_score", "dcrs_score",
]
TARGET = "risk_label"


def load_data():
    df = pd.read_csv(os.path.join(OUT_DIR, "dataset_with_dcrs.csv"))
    features = [c for c in FEATURES if c in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return X, y, features


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else float("nan")
    print(f"\n===== {name} Evaluation =====")
    print(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1 Score : {f1:.4f}\nROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred, zero_division=0))
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": auc}


def train_all_models(X_train, X_test, y_train, y_test):
    results = []

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    if HAS_XGB:
        xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", xgb, X_test, y_test))

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test, y_test))

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate("Decision Tree", dt, X_test, y_test))

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    results.append(evaluate("Support Vector Machine", svm, X_test, y_test))

    models = {"Random Forest": rf, "Logistic Regression": lr, "Decision Tree": dt, "Support Vector Machine": svm}
    if HAS_XGB:
        models["XGBoost"] = xgb
    return models, results


def leakage_check_retrain(X_train, X_test, y_train, y_test):
    """Step 50-51: retrain Random Forest without risk_score_heuristic to check for feature leakage."""
    print("\n=== Step 50-51: Feature-Leakage Check — Retrain Random Forest without risk_score_heuristic ===")
    if "risk_score_heuristic" not in X_train.columns:
        print("[INFO] risk_score_heuristic not present in feature set — skipping leakage check.")
        return None
    cols = [c for c in X_train.columns if c != "risk_score_heuristic"]
    rf_check = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_check.fit(X_train[cols], y_train)
    result = evaluate("Random Forest (leakage-check, no risk_score_heuristic)", rf_check, X_test[cols], y_test)
    return result


def save_comparison(results):
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print("\n===== Machine Learning Model Comparison =====")
    print(comp_df.to_string(index=False))
    return comp_df


if __name__ == "__main__":
    print("=== PHASE 10: MACHINE LEARNING MODEL TRAINING AND COMPARISON ===\n")
    X, y, features = load_data()
    print("Features used:", features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models, results = train_all_models(X_train, X_test, y_train, y_test)
    leakage_result = leakage_check_retrain(X_train, X_test, y_train, y_test)
    if leakage_result:
        results.append(leakage_result)

    comp_df = save_comparison(results)

    # Select best model by F1-score among the five primary classifiers (exclude leakage-check row)
    primary = comp_df[~comp_df["model"].str.contains("leakage-check")]
    best_name = primary.sort_values("f1_score", ascending=False).iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nBest performing model selected: {best_name}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.pkl"))
    print(f"Saved best model -> {MODEL_DIR}/best_model.pkl")

    # Also persist the train/test split so Phase 11 & 13 can reuse the exact same test set
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)
