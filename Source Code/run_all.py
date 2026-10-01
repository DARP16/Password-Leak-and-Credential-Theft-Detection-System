"""
Runs Phases 5-11, 13-14 in sequence (Phase 12 is a Streamlit app and
must be launched separately with `streamlit run phase_12_streamlit_app.py`).
"""
import subprocess
import sys
import os

PHASES = [
    "phase_05_dataset_import.py",
    "phase_06_dataset_cleaning_unification.py",
    "phase_07_bdi.py",
    "phase_08_abfe.py",
    "phase_09_dcrs.py",
    "phase_10_machine_learning.py",
    "phase_11_shap.py",
    "phase_13_evaluation.py",
    "phase_14_finalization.py",
]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    for script in PHASES:
        print(f"\n{'=' * 70}\nRUNNING {script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, os.path.join(here, script)], cwd=project_root)
        if result.returncode != 0:
            print(f"\n[FAILED] {script} exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)
    print("\nAll phases completed successfully.")
    print("To launch the dashboard, run: streamlit run phase_12_streamlit_app.py")
