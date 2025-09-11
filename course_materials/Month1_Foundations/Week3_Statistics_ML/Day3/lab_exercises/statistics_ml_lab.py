"""
Statistics for Machine Learning Lab - Week 3, Day 3

This lab will help you apply statistical concepts to machine learning tasks,
including feature selection, model evaluation, and interpreting results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, accuracy_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Exercise 1: Data Preparation and Exploration
# =============================================================================
print("\n=== Exercise 1: Data Preparation and Exploration ===\n")

# Load the Boston housing dataset (or California housing as Boston is deprecated)
from sklearn.datasets import fetch_california_housing, load_iris

# For regression
housing = fetch_california_housing()
X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
y_reg = housing.target

# For classification
iris = load_iris()
X_clf = pd.DataFrame(iris.data, columns=iris.feature_names)
y_clf = iris.target

print("Regression Dataset (California Housing):")
print(f"Features: {X_reg.shape[1]}, Samples: {X_reg.shape[0]}")
print("\nClassification Dataset (Iris):")
print(f"Features: {X_clf.shape[1]}, Samples: {X_clf.shape[0]}, Classes: {len(np.unique(y_clf))}")

# 1.1 Basic statistics for regression data
print("\nRegression Data Statistics:")
print(X_reg.describe())

# 1.2 Check for missing values
print("\nMissing Values (Regression Data):")
print(X_reg.isnull().sum())

# 1.3 Visualize distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(X_reg.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(X_reg[col], kde=True)
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.savefig('regression_features_distribution.png')
print("\nSaved: regression_features_distribution.png")

# =============================================================================
# Exercise 2: Feature Selection for Regression
# =============================================================================
print("\n=== Exercise 2: Feature Selection for Regression ===\n")

# 2.1 Correlation Analysis
plt.figure(figsize=(10, 8))
corr_matrix = X_reg.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Saved: correlation_matrix.png")

# 2.2 Feature Selection using Statistical Tests
# Select top k features using f_regression
k = 5
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_reg, y_reg)
selected_features = X_reg.columns[selector.get_support()]

print(f"\nTop {k} features selected by f_regression:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature} (F-score: {selector.scores_[selector.get_support()][i-1]:.2f})")

# 2.3 Mutual Information for Regression
mi_scores = mutual_info_regression(X_reg, y_reg)
mi_scores = pd.Series(mi_scores, index=X_reg.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_scores.values, y=mi_scores.index)
plt.title('Feature Importance (Mutual Information)')
plt.tight_layout()
plt.savefig('mutual_info_regression.png')
print("Saved: mutual_info_regression.png")

# =============================================================================
# Exercise 3: Model Evaluation with Statistical Methods
# =============================================================================
print("\n=== Exercise 3: Model Evaluation with Statistical Methods ===\n")

# 3.1 Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 3.2 Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Cross-validation scores
    cv_scores = -cross_val_score(
        model, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error'
    )
    
    results[name] = {
        'RMSE': rmse,
        'CV_RMSE_mean': np.mean(np.sqrt(cv_scores)),
        'CV_RMSE_std': np.std(np.sqrt(cv_scores)),
        'cv_scores': np.sqrt(cv_scores)
    }
    
    print(f"\n{name}:")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  CV RMSE: {results[name]['CV_RMSE_mean']:.4f} ± {results[name]['CV_RMSE_std']:.4f}")

# 3.3 Compare models using paired t-test
model1_scores = results['Linear Regression']['cv_scores']
model2_scores = results['Random Forest']['cv_scores']

t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
print(f"\nPaired t-test between Linear Regression and Random Forest:")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    if np.mean(model1_scores) < np.mean(model2_scores):
        print("Linear Regression performs significantly better than Random Forest")
    else:
        print("Random Forest performs significantly better than Linear Regression")
else:
    print("No significant difference between the models")

# =============================================================================
# Exercise 4: Classification with Statistical Feature Selection
# =============================================================================
print("\n=== Exercise 4: Classification with Statistical Feature Selection ===\n")

# 4.1 Split the classification data
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# 4.2 Feature selection for classification
# ANOVA F-test for feature selection
selector = SelectKBest(score_func=f_classif, k=2)
Xc_selected = selector.fit_transform(Xc_train, yc_train)
selected_features = X_clf.columns[selector.get_support()]

print("Selected features for classification:", list(selected_features))

# 4.3 Train and evaluate classifier
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(Xc_train[selected_features], yc_train)

# Make predictions
yc_pred = classifier.predict(Xc_test[selected_features])
yc_pred_proba = classifier.predict_proba(Xc_test[selected_features])

# Evaluate
accuracy = accuracy_score(yc_test, yc_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(yc_test, yc_pred, target_names=iris.target_names))

# 4.4 Permutation Importance
result = permutation_importance(
    classifier, Xc_test[selected_features], yc_test, n_repeats=10, random_state=42
)

# Create a DataFrame with the results
importances = pd.DataFrame({
    'feature': selected_features,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x='importance_mean', y='feature', data=importances)
plt.title('Permutation Importance')
plt.xlabel('Mean Decrease in Accuracy')
plt.tight_layout()
plt.savefig('permutation_importance.png')
print("Saved: permutation_importance.png")

# =============================================================================
# Exercise 5: Model Diagnostics
# =============================================================================
print("\n=== Exercise 5: Model Diagnostics ===\n")

# 5.1 Check linear regression assumptions
# Fit linear regression model
X_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_const).fit()

# Get predictions and residuals
y_pred = model.predict(X_with_const)
residuals = y_train - y_pred

# 5.2 Plot residuals vs fitted values
plt.figure(figsize=(12, 5))

# Residuals vs Fitted
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')

# Q-Q Plot
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.savefig('regression_diagnostics.png')
print("Saved: regression_diagnostics.png")

# 5.3 Check for multicollinearity using VIF
# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X_reg.columns
vif_data["VIF"] = [variance_inflation_factor(X_reg.values, i) 
                   for i in range(len(X_reg.columns))]

print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values('VIF', ascending=False))

# =============================================================================
# Exercise 6: A/B Testing for Model Comparison
# =============================================================================
print("\n=== Exercise 6: A/B Testing for Model Comparison ===\n")

# Simulate A/B test results
# Model A: 120 conversions out of 1000 users
# Model B: 150 conversions out of 1000 users

# Perform two-proportion z-test
count = [120, 150]  # Number of successes
nobs = [1000, 1000]  # Number of trials
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Model A: {count[0]}/{nobs[0]} = {count[0]/nobs[0]:.2%} conversion rate")
print(f"Model B: {count[1]}/{nobs[1]} = {count[1]/nobs[1]:.2%} conversion rate")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\nThere is a statistically significant difference in conversion rates.")
    if count[1]/nobs[1] > count[0]/nobs[0]:
        print("Model B has a higher conversion rate than Model A.")
    else:
        print("Model A has a higher conversion rate than Model B.")
else:
    print("\nNo statistically significant difference in conversion rates.")

print("\nLab exercises completed!")
