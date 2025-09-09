
# Month 2, Week 1: Linear Regression - Exhaustive Deep Dive

## Regularization (Lasso, Ridge, Elastic Net)

### Theoretical Explanation

Regularization is a technique used to prevent overfitting in machine learning models. It does this by adding a penalty term to the cost function, which discourages the model from learning overly complex patterns.

*   **L1 Regularization (Lasso):** Adds a penalty term proportional to the absolute value of the coefficients. This has the effect of shrinking some of the coefficients to zero, which can be used for feature selection.
*   **L2 Regularization (Ridge):** Adds a penalty term proportional to the square of the coefficients. This has the effect of shrinking all of the coefficients towards zero, but it does not set any of them to exactly zero.
*   **Elastic Net:** A combination of L1 and L2 regularization. It has the benefits of both Lasso and Ridge.

### Code Snippets

**Scikit-Learn:**

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(X, y)

# Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
```

## Closed-Form Solution for Linear Regression (Normal Equation)

### Theoretical Explanation

The Normal Equation is an analytical solution to the linear regression problem. It provides a way to find the optimal values of the coefficients without using an iterative optimization algorithm like gradient descent.

**Equation:**

β = (XᵀX)⁻¹Xᵀy

where:
*   β is the vector of coefficients
*   X is the matrix of input features
*   y is the vector of target values

**Advantages:**
*   No need to choose a learning rate.
*   Guaranteed to find the global minimum.

**Disadvantages:**
*   Computationally expensive for large datasets.
*   Can be slow if the number of features is large.

## Generalized Least Squares (GLS)

### Theoretical Explanation

Generalized Least Squares (GLS) is an extension of Ordinary Least Squares (OLS) that can be used when the errors are not independent or have non-constant variance.

**Assumptions of OLS:**
*   The errors are independent.
*   The errors have constant variance.

**When to use GLS:**
*   When the errors are correlated (e.g., in time series data).
*   When the errors have non-constant variance (e.g., in heteroscedastic data).

## Robust Regression

### Theoretical Explanation

Robust regression is a type of regression that is less sensitive to outliers than OLS. Outliers can have a large influence on the OLS estimates, so robust regression can be a good alternative when there are outliers in the data.

**Methods:**
*   **M-estimators:** A class of robust estimators that are based on minimizing a function of the residuals.
*   **RANSAC (Random Sample Consensus):** An iterative method for estimating the parameters of a model from a set of observed data that contains outliers.

## Interpreting Interaction Terms and Categorical Variables

### Theoretical Explanation

*   **Interaction Terms:** An interaction term is a product of two or more predictor variables. It is used to model the situation where the effect of one predictor variable on the response variable depends on the value of another predictor variable.
*   **Categorical Variables:** A categorical variable is a variable that can take on one of a limited, and usually fixed, number of possible values. To use a categorical variable in a regression model, it must be converted into a set of numerical variables. This can be done using one-hot encoding.
