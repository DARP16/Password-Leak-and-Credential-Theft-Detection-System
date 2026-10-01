# Password Leak and Credential Theft Detection System

A machine learning-based system that detects suspicious login behaviour using behavioural analytics instead of static, rule-based signatures. Built as a Major Project (01CE0716) at Marwadi University, Department of Computer Engineering.

## Overview

Traditional authentication defenses rely on fixed thresholds (e.g., locking an account after N failed attempts) or signature-based intrusion detection. These struggle against distributed, low-and-slow attacks like password spraying. This project instead profiles *how* a login happens — analyzing behavioural and network-level patterns — to classify each attempt as **Low-Risk** or **High-Risk**, and uses SHAP to explain every prediction.

## Key Features

- **Controlled attack simulation** — manual SSH logins, Hydra-based brute-force, and password-spraying attacks against isolated test accounts
- **Multi-source log capture** — Zeek (network), Suricata (IDS/alerts), and Auditd (host-level authentication events)
- **Three custom behavioural risk metrics:**
  - **BDI** – Behaviour Drift Index
  - **ABFE** – Adaptive Behavioural Fingerprint Entropy
  - **DCRS** – Dynamic Credential Risk Score
- **Five ML classifiers compared** — Random Forest, XGBoost, Logistic Regression, Decision Tree, SVM
- **Explainable AI** — SHAP summary and waterfall plots for model interpretability
- **Real-time dashboard** — interactive Streamlit app for live risk scoring

## Pipeline (14 Phases)

| Phase | Description |
|---|---|
| 1 | Environment & Tool Setup |
| 2 | Test User Account Provisioning |
| 3 | Attack Simulation (Hydra brute-force & password spraying) |
| 4 | Log Collection & Monitoring (Zeek, Suricata, Auditd) |
| 5 | Dataset Import |
| 6 | Data Cleaning & Unification |
| 7 | Behaviour Drift Index (BDI) Engineering |
| 8 | Adaptive Behavioural Fingerprint Entropy (ABFE) Engineering |
| 9 | Dynamic Credential Risk Score (DCRS) Engineering |
| 10 | Machine Learning Model Training & Comparison |
| 11 | Explainable AI with SHAP |
| 12 | Streamlit Dashboard Deployment |
| 13 | Performance Evaluation |
| 14 | Conclusion & Future Scope |

Phase-wise implementation screenshots are available in this repository under `PHASE 1/` through `PHASE 14/`.

## Tools & Technologies

- **Attack Simulation:** Hydra
- **Network Monitoring:** Wireshark, Zeek
- **Intrusion Detection:** Suricata
- **Host Auditing:** Auditd
- **Data Processing:** Python, Pandas
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit

## Results

The final Random Forest model was evaluated on a held-out test set of 1,400 login attempts (797 Low-Risk, 603 High-Risk), achieving strong accuracy, precision, recall, and F1-score, with the engineered BDI/ABFE/DCRS features ranking among the most influential predictors per SHAP analysis.

## Team

| Name | Enrolment No. | Contribution |
|---|---|---|
| Darp Kalavadia | 92410103115 | Practical implementation (all 14 phases), presentation (PPT), project report |
| Vasu Vachhani | 92410103087 | Research paper, literature review, research table |
| Om Joshi | 92410103013 | Research paper, literature review, research table |

**Guide:** Prof. Yogeshwar Prajapati, Assistant Professor, Department of Computer Engineering, Marwadi University

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
