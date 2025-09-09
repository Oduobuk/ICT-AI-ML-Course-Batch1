# Month 2, Week 3: Decision Trees and Bagging - Deep Dive

## Decision Tree Splitting Criteria

*   **Impurity Measures:** Decision trees are built by recursively splitting the data into smaller and smaller subsets. The goal of each split is to create subsets that are as pure as possible. Purity is measured using an impurity measure, such as Gini impurity or entropy.
    *   **Gini Impurity:** A measure of the probability of misclassifying a randomly chosen element from a set. A Gini impurity of 0 means that all elements in the set belong to the same class.
    *   **Entropy:** A measure of the amount of uncertainty in a set. An entropy of 0 means that all elements in the set belong to the same class.
*   **Information Gain:** The reduction in impurity that is achieved by splitting a set. The split that results in the greatest information gain is chosen.

## Pruning Techniques

*   **Pruning:** A technique used to prevent overfitting in decision trees. It does this by removing branches from the tree that are not providing much predictive power.
*   **Pre-pruning:** Stopping the growth of the tree early. This can be done by setting a maximum depth for the tree, or by setting a minimum number of samples required to split a node.
*   **Post-pruning:** Growing the tree to its full depth and then removing branches. This is also known as cost-complexity pruning.

## Bagging Algorithm

*   **Bootstrap Aggregating (Bagging):** An ensemble learning technique that is used to reduce the variance of a model. It does this by training multiple models on different bootstrap samples of the training data and then averaging their predictions.
*   **Bootstrap Sampling:** A resampling technique that is used to create multiple datasets from a single dataset. It does this by randomly sampling with replacement from the original dataset.
*   **Out-of-Bag (OOB) Error Estimation:** A way to estimate the generalization error of a bagging model without using a separate test set. It does this by using the instances that were not included in each bootstrap sample to evaluate the model.

## Random Forest Mechanics

*   **Random Forest:** An ensemble learning technique that is an extension of bagging. It is used to further reduce the variance of a model by adding an additional layer of randomness.
*   **Random Feature Subset Selection:** In addition to training each tree on a different bootstrap sample of the data, random forests also randomly select a subset of the features to use for each split. This helps to decorrelate the trees and further reduce the variance of the model.

## Feature Importance

*   **Feature Importance:** A measure of how much each feature contributes to the predictions of a model. It can be used to identify the most important features and to gain a better understanding of the model.
*   **Methods:**
    *   **Mean Decrease in Impurity:** A measure of the total reduction in impurity that is achieved by splitting on a particular feature.
    *   **Permutation Importance:** A measure of the decrease in model performance that is observed when a particular feature is randomly permuted.