"""
Statistics for Machine Learning Lab - Solutions

This file contains the solutions to the Statistics for ML lab exercises.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, f_classif
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from statsmodels.stats.proportion import proportions_ztest

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Exercise 1: Data Preparation and Exploration - SOLUTION
# =============================================================================
print("\n=== Exercise 1: Data Preparation and Exploration - SOLUTION ===\n")

# Load datasets
housing = fetch_california_housing()
X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
y_reg = housing.target

iris = load_iris()
X_clf = pd.DataFrame(iris.data, columns=iris.feature_names)
y_clf = iris.target

print("=== Expected Output ===")
print("Regression Dataset (California Housing):")
print(f"Features: {X_reg.shape[1]}, Samples: {X_reg.shape[0]}")
print("\nClassification Dataset (Iris):")
print(f"Features: {X_clf.shape[1]}, Samples: {X_clf.shape[0]}, Classes: {len(np.unique(y_clf))}")

# =============================================================================
# Exercise 2: Feature Selection for Regression - SOLUTION
# =============================================================================
print("\n=== Exercise 2: Feature Selection for Regression - SOLUTION ===\n")

# 2.2 Feature Selection using f_regression
k = 5
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_reg, y_reg)
selected_features = X_reg.columns[selector.get_support()]

print("=== Expected Output ===")
print(f"\nTop {k} features selected by f_regression:")
for i, (feature, score) in enumerate(zip(selected_features, 
                                      selector.scores_[selector.get_support()]), 1):
    print(f"{i}. {feature} (F-score: {score:.2f})")

# 2.3 Mutual Information for Regression
mi_scores = mutual_info_regression(X_reg, y_reg)
mi_scores = pd.Series(mi_scores, index=X_reg.columns).sort_values(ascending=False)

print("\nTop features by Mutual Information:")
print(mi_scores.head())

# =============================================================================
# Exercise 3: Model Evaluation with Statistical Methods - SOLUTION
# =============================================================================
print("\n=== Exercise 3: Model Evaluation with Statistical Methods - SOLUTION ===\n")

# 3.1-3.2 Train and evaluate models
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    cv_scores = -cross_val_score(
        model, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error'
    )
    results[name] = {
        'RMSE': rmse,
        'CV_RMSE_mean': np.mean(np.sqrt(cv_scores)),
        'CV_RMSE_std': np.std(np.sqrt(cv_scores)),
        'cv_scores': np.sqrt(cv_scores)
    }

print("=== Expected Output ===")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Test RMSE: {metrics['RMSE']:.4f}")
    print(f"  CV RMSE: {metrics['CV_RMSE_mean']:.4f} ± {metrics['CV_RMSE_std']:.4f}")

# 3.3 Compare models using paired t-test
model1_scores = results['Linear Regression']['cv_scores']
model2_scores = results['Random Forest']['cv_scores']

t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
print(f"\nPaired t-test between Linear Regression and Random Forest:")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

# =============================================================================
# Exercise 4: Classification with Statistical Feature Selection - SOLUTION
# =============================================================================
print("\n=== Exercise 4: Classification with Statistical Feature Selection - SOLUTION ===\n")

# 4.2-4.3 Feature selection and model training
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Select top 2 features using f_classif
selector = SelectKBest(score_func=f_classif, k=2)
Xc_selected = selector.fit_transform(Xc_train, yc_train)
selected_features = X_clf.columns[selector.get_support()]

# Train classifier
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(Xc_train[selected_features], yc_train)

# Evaluate
yc_pred = classifier.predict(Xc_test[selected_features])
accuracy = accuracy_score(yc_test, yc_pred)

print("=== Expected Output ===")
print("\nSelected features for classification:", list(selected_features))
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(yc_test, yc_pred, target_names=iris.target_names))

# =============================================================================
# Exercise 5: Model Diagnostics - SOLUTION
# =============================================================================
print("\n=== Exercise 5: Model Diagnostics - SOLUTION ===\n")

# 5.1-5.3 Check linear regression assumptions and VIF
X_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_const).fit()
residuals = y_train - model.predict(X_with_const)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_reg.columns
vif_data["VIF"] = [variance_inflation_factor(X_reg.values, i) 
                   for i in range(len(X_reg.columns))]

print("=== Expected Output ===")
print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values('VIF', ascending=False))

# =============================================================================
# Exercise 6: A/B Testing for Model Comparison - SOLUTION
# =============================================================================
print("\n=== Exercise 6: A/B Testing for Model Comparison - SOLUTION ===\n")

# Simulate A/B test results
count = [120, 150]  # Number of successes
nobs = [1000, 1000]  # Number of trials
z_stat, p_value = proportions_ztest(count, nobs)

print("=== Expected Output ===")
print(f"Model A: {count[0]}/{nobs[0]} = {count[0]/nobs[0]:.2%} conversion rate")
print(f"Model B: {count[1]}/{nobs[1]} = {count[1]/nobs[1]:.2%} conversion rate")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

print("\nAll solutions have been executed. Compare your results with the expected outputs above.")
