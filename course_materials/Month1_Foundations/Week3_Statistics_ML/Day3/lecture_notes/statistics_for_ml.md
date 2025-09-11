# Day 3: Statistical Concepts in Machine Learning

## 📚 Learning Objectives
By the end of this session, you will be able to:
- Understand the role of statistics in machine learning
- Apply statistical tests for feature selection
- Understand the assumptions behind common ML algorithms
- Evaluate models using statistical methods
- Apply statistical techniques to improve model performance

## 🔍 1. Statistics in the ML Pipeline

### 1.1 Data Understanding Phase
- **Descriptive Statistics**: Understanding data distributions, central tendencies, and variability
- **Data Quality Assessment**: Identifying missing values, outliers, and anomalies
- **Exploratory Data Analysis (EDA)**: Visualizing relationships between variables

### 1.2 Data Preprocessing
- **Handling Missing Data**:
  - Deletion methods (listwise, pairwise)
  - Imputation (mean, median, mode, predictive modeling)
  - Indicator variables for missingness

- **Outlier Detection and Treatment**:
  - Z-score method
  - IQR method
  - Robust scaling
  - Winsorization

### 1.3 Feature Engineering
- **Feature Transformation**:
  - Log, square root, Box-Cox transformations
  - Standardization and normalization
  - Encoding categorical variables

- **Feature Creation**:
  - Interaction terms
  - Polynomial features
  - Binning continuous variables

## 📊 2. Statistical Tests for Feature Selection

### 2.1 Correlation Analysis
- **Pearson's r**: Measures linear correlation between continuous variables
  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  # Calculate correlation matrix
  corr_matrix = df.corr()
  
  # Visualize correlation matrix
  plt.figure(figsize=(12, 8))
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
  plt.title('Correlation Matrix')
  plt.tight_layout()
  plt.show()
  ```

- **Spearman's ρ**: Measures monotonic relationships (not necessarily linear)
  ```python
  # Calculate Spearman's rank correlation
  spearman_corr = df.corr(method='spearman')
  ```

### 2.2 Statistical Tests for Feature Selection

#### 2.2.1 Continuous Target Variable
- **ANOVA F-test**: Test if groups have the same population means
  ```python
  from sklearn.feature_selection import f_regression, SelectKBest
  
  # Select top k features using ANOVA F-test
  selector = SelectKBest(score_func=f_regression, k=5)
  X_selected = selector.fit_transform(X, y)
  
  # Get selected feature indices
  selected_features = X.columns[selector.get_support()]
  print(f"Selected features: {list(selected_features)}")
  ```

- **Mutual Information Regression**: Measures dependency between variables
  ```python
  from sklearn.feature_selection import mutual_info_regression
  
  # Calculate mutual information
  mi_scores = mutual_info_regression(X, y)
  mi_scores = pd.Series(mi_scores, index=X.columns)
  mi_scores = mi_scores.sort_values(ascending=False)
  
  # Plot feature importance
  plt.figure(figsize=(10, 6))
  sns.barplot(x=mi_scores.values, y=mi_scores.index)
  plt.title('Feature Importance (Mutual Information)')
  plt.tight_layout()
  plt.show()
  ```

#### 2.2.2 Categorical Target Variable
- **Chi-square test**: Test independence between categorical variables
  ```python
  from sklearn.feature_selection import chi2, SelectKBest
  
  # Select top k features using chi-square test
  selector = SelectKBest(score_func=chi2, k=5)
  X_selected = selector.fit_transform(X, y)
  
  # Get p-values
  p_values = pd.DataFrame({
      'feature': X.columns,
      'p_value': selector.pvalues_
  }).sort_values('p_value')
  
  print("Features sorted by p-value (lower is better):")
  print(p_values)
  ```

- **ANOVA F-test for classification**:
  ```python
  from sklearn.feature_selection import f_classif
  
  # Calculate F-scores and p-values
  f_scores, p_values = f_classif(X, y)
  
  # Create a DataFrame with the results
  anova_results = pd.DataFrame({
      'feature': X.columns,
      'f_score': f_scores,
      'p_value': p_values
  }).sort_values('f_score', ascending=False)
  
  print("ANOVA F-test results:")
  print(anova_results.head())
  ```

## 🎯 3. Statistical Assumptions in ML Models

### 3.1 Linear Regression Assumptions
1. **Linearity**: Relationship between predictors and target is linear
   - **Check**: Plot residuals vs. fitted values
   - **Fix**: Apply transformations (log, square root, etc.)

2. **Independence**: Residuals are independent of each other
   - **Check**: Durbin-Watson test (values 1.5-2.5 suggest no autocorrelation)
   - **Fix**: Use time series models if data is time-dependent

3. **Homoscedasticity**: Constant variance of residuals
   - **Check**: Plot residuals vs. fitted values (look for funnel shape)
   - **Fix**: Transform the target variable or use weighted least squares

4. **Normality of Residuals**: Residuals are normally distributed
   - **Check**: Q-Q plot, Shapiro-Wilk test
   - **Fix**: Transform the target variable

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# Fit linear regression model
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# Get predictions and residuals
y_pred = model.predict(X_with_const)
residuals = y - y_pred

# 1. Check for linearity
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()

# 2. Check for homoscedasticity
_, p_value, _, _ = het_breuschpagan(residuals, X_with_const)
print(f"Breusch-Pagan test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Warning: Heteroscedasticity detected (p < 0.05)")

# 3. Check for normality of residuals
plt.figure(figsize=(10, 5))
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("Warning: Residuals are not normally distributed (p < 0.05)")

# 4. Check for autocorrelation (Durbin-Watson test)
dw = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw:.4f}")
if dw < 1.5 or dw > 2.5:
    print("Warning: Potential autocorrelation detected")
```

### 3.2 Logistic Regression Assumptions
1. **Binary Outcome**: Dependent variable is binary
2. **No Multicollinearity**: Independent variables are not too highly correlated
   - **Check**: Variance Inflation Factor (VIF)
   - **Fix**: Remove highly correlated features or use regularization

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(len(X.columns))]

print("Variance Inflation Factors (VIF):")
print(vif_data.sort_values('VIF', ascending=False))

# Interpretation:
# VIF < 5: No multicollinearity
# 5 ≤ VIF < 10: Moderate multicollinearity
# VIF ≥ 10: High multicollinearity (consider removing the feature)
```

## 📈 4. Model Evaluation with Statistical Methods

### 4.1 Cross-Validation and Statistical Significance
- **k-Fold Cross-Validation**: Estimate model performance
- **Paired t-test**: Compare model performances

```python
from sklearn.model_selection import cross_val_score
from scipy import stats

# Example: Compare two models using cross-validation
model1_scores = cross_val_score(model1, X, y, cv=10, scoring='accuracy')
model2_scores = cross_val_score(model2, X, y, cv=10, scoring='accuracy')

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

print(f"Model 1 mean accuracy: {model1_scores.mean():.4f}")
print(f"Model 2 mean accuracy: {model2_scores.mean():.4f}")
print(f"Paired t-test p-value: {p_value:.4f}")

if p_value < 0.05:
    if model1_scores.mean() > model2_scores.mean():
        print("Model 1 performs significantly better than Model 2")
    else:
        print("Model 2 performs significantly better than Model 1")
else:
    print("No significant difference between the models")
```

### 4.2 Confidence Intervals for Model Performance
```python
import numpy as np
from scipy import stats

def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Calculate confidence interval for model accuracy
mean_acc, lower, upper = mean_confidence_interval(model1_scores)
print(f"Model 1 Accuracy: {mean_acc:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
```

## 🛠️ 5. Practical Applications

### 5.1 A/B Testing for Model Comparison
```python
from statsmodels.stats.proportion import proportions_ztest

# Example: Compare conversion rates between two models
# Model A: 120 conversions out of 1000 users
# Model B: 150 conversions out of 1000 users

# Perform two-proportion z-test
count = [120, 150]  # Number of successes
nobs = [1000, 1000]  # Number of trials
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("There is a statistically significant difference in conversion rates.")
    if count[1]/nobs[1] > count[0]/nobs[0]:
        print("Model B has a higher conversion rate than Model A.")
    else:
        print("Model A has a higher conversion rate than Model B.")
else:
    print("No statistically significant difference in conversion rates.")
```

### 5.2 Feature Importance with Permutation Importance
```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate permutation importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)

# Create a DataFrame with the results
importances = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance_mean', y='feature', data=importances)
plt.title('Permutation Importance')
plt.xlabel('Mean Decrease in Accuracy')
plt.tight_layout()
plt.show()
```

## 📚 6. Key Takeaways

1. **Feature Selection**: Use statistical tests to identify the most relevant features for your model.
2. **Model Assumptions**: Always check the statistical assumptions of your ML models.
3. **Model Evaluation**: Use statistical methods to compare models and estimate performance.
4. **Uncertainty**: Report confidence intervals for model performance metrics.
5. **A/B Testing**: Use statistical tests to compare different models or strategies in production.

## 🔍 7. Further Reading

1. [Scikit-learn Documentation on Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
2. [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
3. [Interpretable Machine Learning by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)
4. [Statistical Methods for Machine Learning](https://machinelearningmastery.com/statistics_for_machine_learning/) by Jason Brownlee
