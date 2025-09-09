# Month 3, Week 1: Clustering

## Lesson Plan

*   **K-Means Clustering:**
    *   **Concept:** Understand how K-Means partitions data into K clusters and methods for determining the optimal number of clusters.
    *   **Hands-on:** Use scikit-learn to build and train a K-Means clustering model. Use the Elbow method and silhouette score to determine the optimal number of clusters.

*   **Hierarchical Clustering:**
    *   **Concept:** Explore a method that builds a hierarchy of clusters, represented by a dendrogram.
    *   **Hands-on:** Use scikit-learn to perform hierarchical clustering. Visualize the dendrogram and experiment with different linkage methods.

*   **DBSCAN:**
    *   **Concept:** Learn about a clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.
    *   **Hands-on:** Use scikit-learn to perform DBSCAN clustering. Analyze its sensitivity to the `eps` and `min_samples` parameters.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 9: Unsupervised Learning Techniques.
    *   **"An Introduction to Statistical Learning, 2nd Edition" by James, Witten, Hastie, and Tibshirani:**
        *   Chapter 12: Unsupervised Learning.

*   **Further Reading:**
    *   **StatQuest with Josh Starmer:**
        *   K-Means Clustering: [https://www.youtube.com/watch?v=4b5d3v_GUFs](https://www.youtube.com/watch?v=4b5d3v_GUFs)
        *   Hierarchical Clustering: [https://www.youtube.com/watch?v=7xHsRkOdVwo](https://www.youtube.com/watch?v=7xHsRkOdVwo)
        *   DBSCAN: [https://www.youtube.com/watch?v=RDZUdRSDpPM](https://www.youtube.com/watch?v=RDZUdRSDpPM)

## Assignments

*   **Assignment 1: K-Means Clustering**
    *   Choose a dataset for clustering (e.g., from Kaggle or the UCI Machine Learning Repository).
    *   Perform exploratory data analysis (EDA).
    *   Build and train a K-Means clustering model using scikit-learn.
    *   Use the Elbow method and silhouette score to determine the optimal number of clusters.
    *   Visualize the clusters.

*   **Assignment 2: Hierarchical Clustering**
    *   Use the dataset from Assignment 1.
    *   Perform hierarchical clustering using scikit-learn.
    *   Visualize the dendrogram and experiment with different linkage methods.

*   **Assignment 3: DBSCAN**
    *   Use the dataset from Assignment 1.
    *   Perform DBSCAN clustering using scikit-learn.
    *   Analyze its sensitivity to the `eps` and `min_samples` parameters.
    *   Compare the results of DBSCAN with the results of K-Means and hierarchical clustering.