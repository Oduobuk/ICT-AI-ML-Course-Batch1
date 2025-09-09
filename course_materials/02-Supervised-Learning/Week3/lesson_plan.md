# Month 2, Week 3: Decision Trees and Bagging

## Lesson Plan

*   **Decision Tree Principles:**
    *   **Concept:** Understand how decision trees make splits based on data characteristics, using impurity measures like Gini impurity and entropy.
    *   **Hands-on:** Use scikit-learn to build and train a decision tree classifier.

*   **Visualizing Trees and Overfitting:**
    *   **Concept:** Learn to interpret tree structures and identify signs of overfitting.
    *   **Hands-on:** Visualize a decision tree and experiment with different tree depths to observe overfitting.

*   **Introduction to Ensemble Learning: Bagging:**
    *   **Concept:** Grasp the concept of combining multiple models to improve performance.
    *   **Hands-on:** Use scikit-learn to build and train a bagging classifier.

*   **Random Forests:**
    *   **Concept:** Explore a powerful ensemble method and how to extract insights from it.
    *   **Hands-on:** Build and train a random forest classifier. Extract and interpret feature importances.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 6: Decision Trees.
        *   Chapter 7: Ensemble Learning and Random Forests.
    *   **"An Introduction to Statistical Learning, 2nd Edition" by James, Witten, Hastie, and Tibshirani:**
        *   Chapter 8: Tree-Based Methods.

*   **Further Reading:**
    *   **StatQuest with Josh Starmer:**
        *   Decision Trees: [https://www.youtube.com/watch?v=7VeUPuFGJHk](https://www.youtube.com/watch?v=7VeUPuFGJHk)
        *   Random Forests: [https://www.youtube.com/watch?v=J4Wdy0Wc_xQ](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

## Assignments

*   **Assignment 1: Decision Tree for Classification**
    *   Choose a classification dataset (e.g., from Kaggle or the UCI Machine Learning Repository).
    *   Perform exploratory data analysis (EDA).
    *   Build and train a decision tree classifier using scikit-learn.
    *   Visualize the tree and interpret the results.

*   **Assignment 2: Overfitting and Pruning**
    *   Use the dataset from Assignment 1.
    *   Train decision trees with different depths and observe the effect on the training and test accuracy.
    *   Use pruning techniques to prevent overfitting.

*   **Assignment 3: Bagging and Random Forests**
    *   Use the dataset from Assignment 1.
    *   Build and train a bagging classifier and a random forest classifier.
    *   Compare the performance of the two models with the performance of a single decision tree.
    *   Extract and interpret the feature importances from the random forest model.