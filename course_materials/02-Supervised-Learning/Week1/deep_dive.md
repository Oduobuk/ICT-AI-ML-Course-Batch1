# Month 2, Week 1: Linear Regression - Deep Dive

## Mathematical Foundations of Linear Regression

*   **Ordinary Least Squares (OLS):** The goal of OLS is to find the line (or hyperplane) that minimizes the sum of the squared differences between the observed values and the values predicted by the model. This is the "least squares" part.
    *   **Example:** Imagine you have a scatter plot of data points. OLS finds the line that is closest to all the points simultaneously.
*   **Assumptions of Linear Regression:**
    1.  **Linearity:** The relationship between the predictors and the outcome is linear.
    2.  **Independence:** The errors are independent of each other.
    3.  **Homoscedasticity:** The errors have constant variance.
    4.  **Normality of Residuals:** The errors are normally distributed.
    *   **Why do they matter?** If these assumptions are violated, the estimates of the coefficients may be biased or inefficient.

## Gradient Descent Variants

*   **Gradient Descent:** An iterative optimization algorithm that is used to find the minimum of a function. It works by taking steps in the direction of the negative gradient of the function.
*   **Batch Gradient Descent:** Computes the gradient of the cost function with respect to the parameters using the entire training dataset. It is slow for large datasets.
*   **Stochastic Gradient Descent (SGD):** Computes the gradient of the cost function with respect to the parameters using a single training example at a time. It is faster than batch gradient descent, but the convergence can be noisy.
*   **Mini-batch Gradient Descent:** A compromise between batch gradient descent and SGD. It computes the gradient of the cost function with respect to the parameters using a small batch of training examples at a time. It is the most common type of gradient descent used in practice.

## Feature Scaling

*   **Why is it important?** If the features have different scales, the gradient descent algorithm may take a long time to converge. This is because the cost function will be elongated, and the gradient will point in a direction that is not optimal.
*   **Methods:**
    *   **Standardization:** Rescales the features to have a mean of 0 and a standard deviation of 1.
    *   **Normalization:** Rescales the features to have a range of [0, 1].

## Bias-Variance Trade-off in Polynomial Regression

*   **Polynomial Regression:** A type of regression that is used to model non-linear relationships between the predictors and the outcome. It does this by adding polynomial terms to the regression equation.
*   **Bias-Variance Trade-off:**
    *   **High Bias (Underfitting):** A model with high bias is too simple and does not capture the underlying trend in the data. This results in a high error on both the training and test sets.
    *   **High Variance (Overfitting):** A model with high variance is too complex and fits the training data too well. This results in a low error on the training set but a high error on the test set.
    *   **The Goal:** To find a model that has a low bias and a low variance.