# Fair Credit Risk Assessment Using the German Credit Dataset

**Course:** M515A — Ethical Issues for AI | **Author:** Budike Meeraj Kumar | **GH Number:** GH1040835

---

## Overview
This project builds a bias-aware credit risk classification pipeline on the German Credit Dataset (1,000 applicants). It trains two ML models, measures gender-based fairness gaps, and applies post-processing threshold adjustment to produce fairer loan approval decisions — without sacrificing meaningful predictive accuracy.

---

## Dataset
**Source:** [German Credit Data — Kaggle](https://www.kaggle.com/datasets/uciml/german-credit/data)

| Feature | Description |
|---|---|
| Age | Applicant age |
| Sex | Gender (sensitive attribute) |
| Job | Employment skill level (0–3) |
| Housing | own / rent / free |
| Saving accounts | Balance tier (183 missing) |
| Checking account | Balance tier (394 missing) |
| Credit amount | Loan amount in DM |
| Duration | Loan term in months |
| Purpose | Reason for the loan |

**Target:** `Risk` — binary (1 = good credit, 0 = bad credit), engineered via rule-based proxy (high credit amount + long duration + poor savings = bad risk).

---

## Pipeline
```
Raw Data → EDA → Target Engineering → Preprocessing →
Train LR & RF → Measure Bias → Threshold Adjustment → Fair Predictions
```

---

## Notebook Sections

| # | Section |
|---|---|
| 1 | Problem Statement & Business Context |
| 2 | Required Libraries |
| 3 | Load & Explore the Dataset |
| 4 | Exploratory Data Analysis |
| 5 | Target Variable Engineering |
| 6 | Preprocessing |
| 7 | Train/Test Split |
| 8 | Logistic Regression |
| 9 | Random Forest |
| 10 | Bias / Fairness Measurement (pre-correction) |
| 11 | Post-processing Threshold Adjustment |
| 12 | Fairness Measurement (post-correction) |
| 13 | Feature Importance Analysis |
| 14 | Final Discussion & Recommendations |

---

## Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Higher-accuracy ensemble |

---

## Fairness Metrics
- **Demographic Parity Difference (DP Diff):** gap in approval rates between male and female applicants
- **Equal Opportunity Difference:** gap in true positive rates between groups
- **Target:** DP Diff ≤ 0.05 after correction

---

## Key Results
- Checking account balance was the dominant feature (importance ≈ 0.71)
- Post-processing threshold correction substantially reduced gender bias with only –0.33% accuracy cost
- **Recommended model:** Random Forest with per-group threshold correction

---

## Business Recommendations
1. Deploy **Random Forest with threshold correction** as primary model
2. Adopt a **Fairness SLA** of DP Diff ≤ 0.05 for all production updates
3. Run **quarterly bias audits** as the applicant pool evolves
4. Investigate proxy variables (Housing, Purpose) for indirect gender encoding
5. Add richer financial behaviour data to reduce reliance on demographic proxies
6. Apply **human review** for borderline probability scores in [0.40, 0.60]

---

## Tech Stack
Python 3.12 · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

**Google Colab:** Upload `AI_Ethics.ipynb` → upload `german_credit_data.csv` when prompted → Run all cells

**Local:**
```bash
git clone https://github.com/<your-username>/fair-credit-assessment.git
cd fair-credit-assessment
pip install numpy pandas matplotlib seaborn scikit-learn notebook
jupyter notebook AI_Ethics.ipynb
```

---

## Repository Structure
```
fair-credit-assessment/
├── AI_Ethics.ipynb
├── README.md
└── data/
    └── german_credit_data.csv   # Download from Kaggle — not included
```

---

## References
- Hardt et al. (2016). Equality of opportunity in supervised learning. *NeurIPS*
- Feldman et al. (2015). Certifying and removing disparate impact. *KDD*
- European Parliament (2024). EU Artificial Intelligence Act — High-Risk AI Systems
