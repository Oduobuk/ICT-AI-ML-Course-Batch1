# Month 3, Week 3: Model Validation and Hyperparameter Tuning - Deep Dive

## Bias-Variance Decomposition

*   **Bias:** The error that is introduced by approximating a real-world problem with a simplified model. A model with high bias is said to be underfitting.
*   **Variance:** The error that is introduced by the model's sensitivity to small fluctuations in the training data. A model with high variance is said to be overfitting.
*   **Irreducible Error:** The error that is inherent in the data and cannot be reduced by any model.

**The Goal:** To find a model that has a low bias and a low variance.

## Cross-Validation Strategies

*   **Cross-Validation:** A technique for evaluating the performance of a model and for tuning its hyperparameters. It works by splitting the data into a number of folds, and then training and evaluating the model on each fold.
*   **K-Fold Cross-Validation:** A cross-validation technique where the data is split into k folds. The model is then trained on k-1 folds and evaluated on the remaining fold. This process is repeated k times, with each fold being used as the test set once.
*   **Stratified K-Fold Cross-Validation:** A variation of k-fold cross-validation that is used for imbalanced datasets. It ensures that each fold has the same proportion of each class as the original dataset.
*   **Leave-One-Out Cross-Validation (LOOCV):** A cross-validation technique where the number of folds is equal to the number of instances in the data. This is the most computationally expensive type of cross-validation, but it can be useful for small datasets.
*   **Time Series Cross-Validation:** A cross-validation technique that is used for time series data. It ensures that the model is not trained on data from the future.

## Hyperparameter Optimization Algorithms

*   **Grid Search:** A hyperparameter optimization algorithm that exhaustively searches through a grid of hyperparameters.
*   **Random Search:** A hyperparameter optimization algorithm that randomly samples from a distribution of hyperparameters.
*   **Bayesian Optimization:** A hyperparameter optimization algorithm that uses a probabilistic model to select the next set of hyperparameters to evaluate.
*   **Genetic Algorithms:** A hyperparameter optimization algorithm that is inspired by the process of natural selection.

## MLflow Components

*   **MLflow Tracking:** A component of MLflow that is used to log parameters, metrics, and artifacts from machine learning experiments.
*   **MLflow Projects:** A component of MLflow that is used to package code for machine learning experiments.
*   **MLflow Models:** A component of MLflow that is used to standardize the format of machine learning models.
*   **MLflow Model Registry:** A component of MLflow that is used to manage the lifecycle of machine learning models.

## Early Stopping

*   **Early Stopping:** A regularization technique that is used to prevent overfitting. It works by monitoring the performance of the model on a validation set and then stopping the training process when the performance on the validation set starts to degrade.