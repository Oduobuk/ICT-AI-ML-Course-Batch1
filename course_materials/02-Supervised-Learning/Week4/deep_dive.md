# Month 2, Week 4: Boosting and SVM - Deep Dive

## Boosting Algorithms

*   **Sequential Nature of Boosting:** Unlike bagging, which trains models in parallel, boosting trains models sequentially. Each new model is trained to correct the errors of the previous models.
*   **AdaBoost (Adaptive Boosting):**
    *   **Reweighting Misclassified Samples:** AdaBoost works by assigning weights to each training instance. Initially, all weights are equal. At each iteration, the weights of the misclassified instances are increased. This forces the subsequent weak learners to focus on the instances that are difficult to classify.
*   **Gradient Boosting:**
    *   **Predicting Residuals:** Gradient Boosting works by fitting a new weak learner to the residuals of the previous weak learner. The residuals are the difference between the actual values and the predicted values. This allows the new weak learner to correct the errors of the previous weak learner.

## XGBoost/LightGBM Internals

*   **XGBoost (Extreme Gradient Boosting):**
    *   **Optimizations:** XGBoost uses a number of techniques to improve performance, including:
        *   **Parallelization:** XGBoost can train multiple trees in parallel.
        *   **Tree Pruning:** XGBoost uses a more sophisticated tree pruning algorithm than traditional gradient boosting.
        *   **Handling Missing Values:** XGBoost can handle missing values automatically.
        *   **Regularization:** XGBoost includes L1 and L2 regularization to prevent overfitting.
*   **LightGBM (Light Gradient Boosting Machine):**
    *   **Optimizations:** LightGBM is designed to be fast and efficient. It uses a number of techniques to achieve this, including:
        *   **Gradient-based One-Side Sampling (GOSS):** A technique for sampling the data that gives more weight to the instances with larger gradients.
        *   **Exclusive Feature Bundling (EFB):** A technique for bundling mutually exclusive features together to reduce the number of features.

## Support Vector Machines (SVM) Optimization

*   **Maximizing the Margin:** The goal of SVM is to find the hyperplane that best separates the data into different classes. The best hyperplane is the one that maximizes the margin, which is the distance between the hyperplane and the closest data points.
*   **Support Vectors:** The data points that are closest to the hyperplane are called support vectors. These are the data points that are most important for defining the hyperplane.

## Kernel Trick

*   **Handling Non-linearly Separable Data:** The kernel trick is a technique that is used to handle non-linearly separable data. It works by implicitly mapping the data into a higher-dimensional space where it is linearly separable.
*   **Kernel Functions:**
    *   **Polynomial Kernel:** A kernel function that is based on the polynomial function. It can be used to model non-linear relationships of a specific degree.
    *   **RBF Kernel (Radial Basis Function Kernel):** A kernel function that is based on the Gaussian function. It can be used to model complex non-linear relationships.
    *   **Sigmoid Kernel:** A kernel function that is based on the sigmoid function. It is often used in neural networks.

## Soft Margin SVM

*   **Handling Noisy Data:** A soft margin SVM is a variation of SVM that allows for some misclassifications. This can be useful for handling noisy data and for improving the generalization of the model.
*   **Regularization Parameter C:** The regularization parameter C controls the trade-off between maximizing the margin and minimizing the number of misclassifications.
    *   **Small C:** A small value of C will result in a wider margin and more misclassifications. This can be useful for handling noisy data.
    *   **Large C:** A large value of C will result in a narrower margin and fewer misclassifications. This can be useful for data that is well-separated.