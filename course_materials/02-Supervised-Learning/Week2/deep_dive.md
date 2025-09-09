# Month 2, Week 2: Logistic Regression and KNN - Deep Dive

## Logistic Regression

*   **Probabilistic Interpretation:** Logistic regression models the probability that an instance belongs to a particular class. It does this by taking a linear combination of the features and passing it through the sigmoid function.
    *   **Sigmoid Function:** The sigmoid function is an S-shaped function that maps any real value to a value between 0 and 1. This is what allows us to interpret the output of the model as a probability.
*   **Maximum Likelihood Estimation:** The model is trained by finding the parameters that maximize the likelihood of the observed data.

## Decision Boundary

*   **Linear Decision Boundary:** Logistic regression creates a linear decision boundary in the feature space. This means that it can only be used to separate data that is linearly separable.
    *   **Example:** If you have two classes of data that can be separated by a straight line, then logistic regression will be able to learn a model that can accurately classify the data.
*   **Non-linear Decision Boundary:** If the data is not linearly separable, then logistic regression will not be able to learn a model that can accurately classify the data. In this case, you will need to use a more powerful model, such as a support vector machine or a neural network.

## Evaluation Metrics

*   **Accuracy:** The percentage of correctly classified instances.
*   **Precision:** The percentage of positive predictions that are correct.
*   **Recall:** The percentage of positive instances that are correctly classified.
*   **F1-score:** The harmonic mean of precision and recall.
*   **ROC Curve:** A plot of the true positive rate against the false positive rate at various threshold settings.
*   **AUC:** The area under the ROC curve. It is a measure of the overall performance of a classifier.

## K-Nearest Neighbors (KNN)

*   **Concept:** KNN is a non-parametric, instance-based learning algorithm. It is a simple but powerful algorithm that can be used for both classification and regression.
*   **How it works:** To classify a new instance, KNN finds the k nearest neighbors to the instance in the training data and then assigns the new instance to the class that is most common among its neighbors.
*   **'k' and the Bias-Variance Trade-off:**
    *   **Small 'k':** A small value of 'k' will result in a model with a low bias and a high variance. This is because the model will be very sensitive to the noise in the training data.
    *   **Large 'k':** A large value of 'k' will result in a model with a high bias and a low variance. This is because the model will be less sensitive to the noise in the training data.
*   **Distance Metrics:**
    *   **Euclidean Distance:** The straight-line distance between two points.
    *   **Manhattan Distance:** The sum of the absolute differences between the coordinates of two points.
*   **Computational Complexity:** The computational complexity of KNN is high, especially for large datasets. This is because it needs to compute the distance between the new instance and all of the instances in the training data.