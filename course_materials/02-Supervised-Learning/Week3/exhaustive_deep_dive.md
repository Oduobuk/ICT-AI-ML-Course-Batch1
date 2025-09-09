# Month 2, Week 3: Decision Trees and Bagging - Exhaustive Deep Dive

## Generalized Linear Models (GLMs) and Logistic Regression

### Theoretical Explanation

Generalized Linear Models (GLMs) are a class of models that are a generalization of linear regression. They are used to model a wide variety of data, including count data, binary data, and continuous data with non-constant variance.

**Components of a GLM:**
*   **Random Component:** The probability distribution of the response variable.
*   **Systematic Component:** The linear combination of the predictor variables.
*   **Link Function:** A function that links the expected value of the response variable to the systematic component.

**Logistic Regression as a GLM:**
*   **Random Component:** Bernoulli distribution.
*   **Systematic Component:** Linear combination of the predictor variables.
*   **Link Function:** Logit function.

## Regularization in Logistic Regression (L1, L2, Elastic Net)

### Theoretical Explanation

Regularization is a technique used to prevent overfitting in logistic regression models. It does this by adding a penalty term to the cost function, which discourages the model from learning overly complex patterns.

*   **L1 Regularization (Lasso):** Adds a penalty term proportional to the absolute value of the coefficients. This has the effect of shrinking some of the coefficients to zero, which can be used for feature selection.
*   **L2 Regularization (Ridge):** Adds a penalty term proportional to the square of the coefficients. This has the effect of shrinking all of the coefficients towards zero, but it does not set any of them to exactly zero.
*   **Elastic Net:** A combination of L1 and L2 regularization. It has the benefits of both Lasso and Ridge.

### Code Snippets

**Scikit-Learn:**

```python
from sklearn.linear_model import LogisticRegression

# L1 Regularization
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear')
log_reg_l1.fit(X, y)

# L2 Regularization
log_reg_l2 = LogisticRegression(penalty='l2')
log_reg_l2.fit(X, y)

# Elastic Net
log_reg_elastic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
log_reg_elastic.fit(X, y)
```

## Multinomial Logistic Regression (Softmax Regression)

### Theoretical Explanation

Multinomial logistic regression is an extension of logistic regression that is used for multi-class classification problems. It is also known as softmax regression.

**Softmax Function:**

The softmax function is a generalization of the sigmoid function that is used to compute the probabilities of each class. It takes a vector of scores as input and outputs a vector of probabilities that sum to 1.

## Kernelized KNN and Metric Learning

### Theoretical Explanation

*   **Kernelized KNN:** A variation of KNN that uses a kernel function to compute the distances between data points. This can be useful for handling non-linearly separable data.
*   **Metric Learning:** A technique for learning a distance metric that is tailored to the specific dataset. This can be used to improve the performance of KNN.

## Curse of Dimensionality in KNN and Solutions

### Theoretical Explanation

The curse of dimensionality is a phenomenon that occurs when working with high-dimensional data. It refers to the fact that the volume of the feature space increases exponentially with the number of dimensions. This can make it difficult to find nearest neighbors, as all data points tend to be far away from each other.

**Solutions:**
*   **Dimensionality Reduction:** A technique for reducing the number of dimensions in the data. This can be done using techniques like Principal Component Analysis (PCA) or t-SNE.
*   **Feature Selection:** A technique for selecting a subset of the most important features. This can be done using techniques like recursive feature elimination or forward selection.