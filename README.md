# Covid-19 Risk Prediction using Cost-Sensitive ML

This project builds a clinically-aware machine learning pipeline to identify likely COVID-19 patients using routine hospital data. It prioritizes minimizing false negatives — a critical metric in pandemic response — through custom cost-sensitive evaluation and threshold tuning.

## Features

- Logistic Regression, KNN, and Random Forest models
- Custom 2×2 cost matrix to penalize false negatives more than false positives
- Decision threshold tuning (`τ = 0.3`) for recall optimization
- Achieved 99.77% sensitivity by reducing FN from 15 to 3
- Modular, testable code with explainable outputs

## Tech Stack

- Python (Scikit-learn, Pandas)
- MLflow (for tracking models)
- Jupyter Notebooks (for analysis)
- Docker-ready (optional)

## Cost Matrix

```text
[ TP = +2    | FN = -1.5 ]
[ FP = +0.2  | TN = +1   ]
