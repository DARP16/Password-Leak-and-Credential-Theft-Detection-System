"""
PHASE 11 - EXPLAINABLE AI WITH SHAP
Builds a SHAP TreeExplainer on the trained Random Forest model and
the held-out test set, producing both global explanations (summary
plot, bar feature-importance plot) and a local explanation (waterfall
plot) for an individual High-Risk prediction, matching the figures
presented in the project report (Fig. 4.27-4.28).
"""
import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model_and_data():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_list.pkl"))
    X_test = pd.read_csv(os.path.join(OUT_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(OUT_DIR, "y_test.csv")).squeeze("columns")
    print("Step 65 - Loaded model and test dataset for SHAP:", X_test.shape)
    return model, X_test[features], y_test


def build_explainer(model, X_test):
    print("Step 66 - Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def save_summary_plot(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    path = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Step 67 - SHAP summary plot saved -> {path}")


def save_feature_importance_plot(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    path = os.path.join(OUT_DIR, "shap_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Step 68 - SHAP feature-importance plot saved -> {path}")


def save_waterfall_plot(shap_values, y_test, sample_index=None):
    """Step 69: local explanation for a single (preferably High-Risk) prediction."""
    if sample_index is None:
        high_risk_idx = y_test[y_test == 1].index
        sample_index = int(high_risk_idx[0]) if len(high_risk_idx) else 0

    sv = shap_values[sample_index]
    # For binary classifiers, TreeExplainer may return a (features, classes) matrix
    # per sample; select the positive ("High-Risk") class slice for the waterfall plot.
    if len(sv.shape) > 1:
        sv = sv[:, 1]

    plt.figure()
    shap.plots.waterfall(sv, show=False)
    path = os.path.join(OUT_DIR, "shap_waterfall.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Step 69 - SHAP waterfall plot (sample #{sample_index}) saved -> {path}")


if __name__ == "__main__":
    print("=== PHASE 11: EXPLAINABLE AI WITH SHAP ===\n")
    model, X_test, y_test = load_model_and_data()
    explainer, shap_values = build_explainer(model, X_test)

    save_summary_plot(shap_values, X_test)
    save_feature_importance_plot(shap_values, X_test)
    save_waterfall_plot(shap_values, y_test)

    print("\nStep 70 - All SHAP plots generated and saved successfully.")
