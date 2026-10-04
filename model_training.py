"""
Week 4 Task: Supervised Learning Model Implementation
Problem: Binary classification - predict Titanic passenger survival
Dataset: Cleaned Titanic dataset (from Week 1)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate,
                                      GridSearchCV, learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. PROBLEM DEFINITION & DATA LOADING
#    Binary classification: predict `survived` (0/1) for each Titanic
#    passenger from their demographic and ticket attributes.
# ------------------------------------------------------------------
df = pd.read_csv('titanic_cleaned.csv')
print("Loaded:", df.shape)
print(df.dtypes)

# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING / SELECTION
#    Build the model feature set from the already-cleaned Week 1 data.
#    - Drop leakage-prone / redundant columns: 'who' and 'adult_male'
#      are derived directly from sex+age and would leak a near-perfect
#      proxy of the target construction process; 'alone' duplicates
#      is_alone; 'fare' and 'fare_capped' duplicate fare_log (kept as
#      the single fare representation to avoid multicollinearity).
# ------------------------------------------------------------------
drop_cols = ['who', 'adult_male', 'alone', 'fare', 'fare_capped']
model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

categorical_cols = ['sex', 'embarked']
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

X = model_df.drop(columns=['survived'])
y = model_df['survived']
print("\nFinal feature set:", list(X.columns))
print("Target distribution:\n", y.value_counts(normalize=True).round(3))

# ------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
#    Split BEFORE any scaling, to prevent data leakage from the test
#    set influencing the training-time feature distribution - a
#    stricter standard than the Week 1 preprocessing (which scaled
#    the full dataset once, appropriately for descriptive purposes,
#    but not appropriate for a predictive-modeling evaluation).
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
print("Train target balance:\n", y_train.value_counts(normalize=True).round(3))
print("Test target balance:\n", y_test.value_counts(normalize=True).round(3))

scale_cols = ['age', 'fare_log', 'family_size']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])  # transform only, fit on train

# ------------------------------------------------------------------
# 4. MODEL COMPARISON VIA 5-FOLD STRATIFIED CROSS-VALIDATION
#    Three candidate models are compared: Logistic Regression (simple,
#    interpretable baseline), Random Forest (non-linear, handles
#    feature interactions, gives feature importances), and Gradient
#    Boosting (often strong on tabular data of this size).
# ------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(probability=True, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = {}
for name, model in models.items():
    res = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring)
    cv_results[name] = {m: res[f'test_{m}'].mean() for m in scoring}
    cv_results[name]['std_accuracy'] = res['test_accuracy'].std()

cv_results_df = pd.DataFrame(cv_results).T
print("\n5-fold CV results (mean scores on training data):\n", cv_results_df.round(3))

plt.figure(figsize=(9.5, 5))
plot_df = cv_results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].reset_index().melt(
    id_vars='index', var_name='Metric', value_name='Score')
sns.barplot(data=plot_df, x='index', y='Score', hue='Metric', palette='Set2')
plt.title('5-Fold Cross-Validation: Model Comparison')
plt.xlabel('Model')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)
plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_01_model_comparison.png')
plt.close()

# ------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING (Random Forest selected as the primary
#    model going forward: best balance of CV performance and
#    interpretability via feature importances)
# ------------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [4, 6, 8, None],
    'min_samples_leaf': [1, 2, 4],
}
grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid,
                     cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("\nBest Random Forest params:", grid.best_params_)
print("Best CV F1 score:", round(grid.best_score_, 3))

best_rf = grid.best_estimator_

# Validation curve: effect of max_depth specifically, holding other
# tuned params fixed, to visualize the bias/variance trade-off.
depth_range = [2, 3, 4, 5, 6, 8, 10, 12, None]
depth_labels = [str(d) for d in depth_range]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],
                            min_samples_leaf=grid.best_params_['min_samples_leaf'],
                            random_state=RANDOM_STATE),
    X_train_scaled, y_train, param_name='max_depth', param_range=depth_range,
    cv=cv, scoring='f1', n_jobs=-1)

plt.figure(figsize=(8, 4.5))
x_pos = np.arange(len(depth_range))
plt.plot(x_pos, train_scores.mean(axis=1), marker='o', label='Training score', color='#2980b9')
plt.fill_between(x_pos, train_scores.mean(axis=1) - train_scores.std(axis=1),
                  train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='#2980b9')
plt.plot(x_pos, val_scores.mean(axis=1), marker='o', label='Cross-validation score', color='#e67e22')
plt.fill_between(x_pos, val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color='#e67e22')
plt.xticks(x_pos, depth_labels)
plt.title('Validation Curve: Random Forest max_depth vs. F1 Score')
plt.xlabel('max_depth')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig('chart_02_validation_curve.png')
plt.close()

# ------------------------------------------------------------------
# 6. LEARNING CURVE for the tuned Random Forest (diagnoses whether
#    more training data would likely help, and checks for overfitting)
# ------------------------------------------------------------------
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    best_rf, X_train_scaled, y_train, cv=cv, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

plt.figure(figsize=(8, 4.5))
plt.plot(train_sizes, train_scores_lc.mean(axis=1), marker='o', label='Training score', color='#2980b9')
plt.plot(train_sizes, val_scores_lc.mean(axis=1), marker='o', label='Cross-validation score', color='#e67e22')
plt.fill_between(train_sizes, train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
                  train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1), alpha=0.15, color='#2980b9')
plt.fill_between(train_sizes, val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
                  val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1), alpha=0.15, color='#e67e22')
plt.title('Learning Curve: Tuned Random Forest')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig('chart_03_learning_curve.png')
plt.close()

# ------------------------------------------------------------------
# 7. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ------------------------------------------------------------------
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- FINAL TEST SET PERFORMANCE (Tuned Random Forest) ---")
print(f"Accuracy:  {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1 Score:  {test_f1:.3f}")
print(f"ROC-AUC:   {test_roc_auc:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=['Did not survive', 'Survived']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5.5, 4.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Did not survive', 'Survived'],
            yticklabels=['Did not survive', 'Survived'])
plt.title('Confusion Matrix - Tuned Random Forest (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('chart_04_confusion_matrix.png')
plt.close()

# ROC curve - compare tuned RF against the Logistic Regression baseline
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {test_roc_auc:.3f})', color='#27ae60', linewidth=2)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_lr):.3f})',
         color='#2980b9', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.5)')
plt.title('ROC Curve - Test Set')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('chart_05_roc_curve.png')
plt.close()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6.5, 5.5))
plt.plot(rec, prec, color='#8e44ad', linewidth=2, label='Random Forest')
plt.axhline(y_test.mean(), linestyle='--', color='gray', label=f'Baseline (prevalence = {y_test.mean():.2f})')
plt.title('Precision-Recall Curve - Test Set')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig('chart_06_precision_recall_curve.png')
plt.close()

# ------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest)
# ------------------------------------------------------------------
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importances.round(3))

plt.figure(figsize=(8, 5.5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette='viridis', legend=False)
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance (Mean Decrease in Impurity)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('chart_07_feature_importance.png')
plt.close()

# ------------------------------------------------------------------
# 9. LOGISTIC REGRESSION COEFFICIENTS (interpretability comparison)
# ------------------------------------------------------------------
coefs = pd.Series(log_reg.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(8, 5.5))
colors = ['#c0392b' if v < 0 else '#27ae60' for v in coefs.values]
plt.barh(coefs.index, coefs.values, color=colors)
plt.title('Logistic Regression Coefficients\n(positive = increases survival odds)')
plt.xlabel('Coefficient (standardized features)')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('chart_08_logreg_coefficients.png')
plt.close()

# ------------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------------
results_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Tuned Random Forest'],
    'Test_Accuracy': [accuracy_score(y_test, log_reg.predict(X_test_scaled)), test_accuracy],
    'Test_Precision': [precision_score(y_test, log_reg.predict(X_test_scaled)), test_precision],
    'Test_Recall': [recall_score(y_test, log_reg.predict(X_test_scaled)), test_recall],
    'Test_F1': [f1_score(y_test, log_reg.predict(X_test_scaled)), test_f1],
    'Test_ROC_AUC': [roc_auc_score(y_test, y_proba_lr), test_roc_auc],
})
results_summary.to_csv('final_model_results.csv', index=False)
cv_results_df.to_csv('cv_model_comparison.csv')
importances.to_csv('feature_importances.csv')

print("\nAll charts and output files saved. Modeling pipeline complete.")
