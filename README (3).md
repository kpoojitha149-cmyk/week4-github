# Week 4 Task: Supervised Learning Model Implementation

## Overview
This project implements and evaluates a binary classification model predicting
Titanic passenger survival, using the cleaned dataset produced in Week 1 and
building on the exploratory findings from Week 2.

## Contents
| File | Description |
|---|---|
| `model_training.py` | Full pipeline: feature engineering, train/test split, 4-model CV comparison, hyperparameter tuning, final evaluation, all 8 figures |
| `titanic_cleaned.csv` | Input dataset (cleaned in Week 1) |
| `final_model_results.csv` | Test-set metrics for Logistic Regression and the tuned Random Forest |
| `cv_model_comparison.csv` | 5-fold CV metrics for all four candidate models |
| `feature_importances.csv` | Random Forest feature importance values |
| `chart_01...chart_08.png` | All 8 generated figures |
| `Week4_Supervised_Learning_Report.docx` | Full written report |

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python model_training.py
```

## Key results
- 4 models compared via 5-fold stratified cross-validation: Logistic Regression,
  Random Forest, Gradient Boosting, SVM (RBF) - all within 80.2%-82.2% CV accuracy
- Random Forest tuned via GridSearchCV (best F1 = 0.771)
- Final test-set performance: 77.7% accuracy, 0.842 ROC-AUC
- Top features: sex (0.302), fare (0.203), age (0.184) - confirming Week 2's EDA
- Logistic Regression achieved a nearly identical ROC-AUC (0.841) to the tuned
  Random Forest, a notable finding discussed in the report

Full methodology, validation curves, and limitations discussion are in
`Week4_Supervised_Learning_Report.docx`.
