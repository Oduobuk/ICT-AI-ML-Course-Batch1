# Month 2, Week 4: Boosting and Support Vector Machines (SVM)

## Lesson Plan

*   **Ensemble Learning: Boosting:**
    *   **Concept:** Understand how boosting sequentially builds models to correct the errors of previous models. Learn the difference between AdaBoost and Gradient Boosting.
    *   **Hands-on:** Use scikit-learn to build and train a Gradient Boosting classifier.

*   **XGBoost/LightGBM:**
    *   **Concept:** Learn to use powerful gradient boosting frameworks for high-performance machine learning.
    *   **Hands-on:** Use XGBoost or LightGBM to build and train a model on a large dataset. Tune hyperparameters and analyze feature importance.

*   **Support Vector Machines (SVM):**
    *   **Concept:** Explore a robust classification algorithm that finds the optimal separating hyperplane. Understand the concepts of hyperplanes, margins, and support vectors.
    *   **Hands-on:** Use scikit-learn to build and train a linear SVM classifier.

*   **The Kernel Trick:**
    *   **Concept:** Learn how to use the kernel trick to handle non-linearly separable data.
    *   **Hands-on:** Use a non-linear SVM with a kernel (e.g., RBF) to classify a non-linearly separable dataset.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 7: Ensemble Learning and Random Forests (focus on Boosting).
        *   Chapter 5: Support Vector Machines.
    *   **"An Introduction to Statistical Learning, 2nd Edition" by James, Witten, Hastie, and Tibshirani:**
        *   Chapter 8: Tree-Based Methods (focus on Boosting).
        *   Chapter 9: Support Vector Machines.

*   **Further Reading:**
    *   **StatQuest with Josh Starmer:**
        *   Gradient Boost: [https://www.youtube.com/watch?v=3CC4N4z3GJc](https://www.youtube.com/watch?v=3CC4N4z3GJc)
        *   Support Vector Machines: [https://www.youtube.com/watch?v=efR1C6CvhmE](https://www.youtube.com/watch?v=efR1C6CvhmE)

## Assignments

*   **Assignment 1: Gradient Boosting**
    *   Choose a classification or regression dataset.
    *   Build and train a Gradient Boosting model using scikit-learn.
    *   Tune the hyperparameters of the model to improve its performance.

*   **Assignment 2: XGBoost/LightGBM**
    *   Use the dataset from Assignment 1.
    *   Build and train a model using XGBoost or LightGBM.
    *   Compare the performance of the XGBoost/LightGBM model with the performance of the Gradient Boosting model from Assignment 1.

*   **Assignment 3: Support Vector Machines**
    *   Generate a non-linearly separable dataset.
    *   Build and train a linear SVM and a non-linear SVM with an RBF kernel.
    *   Visualize the decision boundaries of the two models and compare their performance.