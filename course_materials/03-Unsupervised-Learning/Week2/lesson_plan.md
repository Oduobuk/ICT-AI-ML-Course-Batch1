# Month 3, Week 2: Dimensionality Reduction

## Lesson Plan

*   **Principal Component Analysis (PCA):**
    *   **Concept:** Understand how PCA transforms data into a new set of orthogonal variables (principal components) that capture the most variance.
    *   **Hands-on:** Use scikit-learn to perform PCA on a high-dimensional dataset. Determine the optimal number of components to retain and visualize the data in reduced dimensions.

*   **t-SNE and UMAP:**
    *   **Concept:** Explore non-linear dimensionality reduction techniques specifically designed for visualizing complex datasets in 2D or 3D.
    *   **Hands-on:** Use t-SNE and UMAP to visualize a complex dataset (e.g., MNIST digits or a text embedding dataset). Compare the resulting visualizations.

*   **Feature Selection vs. Feature Extraction:**
    *   **Concept:** Differentiate between selecting a subset of original features and creating new, transformed features.

## Reading Assignments

*   **Primary Reading:**
    *   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd Edition" by Aurélien Géron:**
        *   Chapter 8: Dimensionality Reduction.
    *   **"An Introduction to Statistical Learning, 2nd Edition" by James, Witten, Hastie, and Tibshirani:**
        *   Chapter 10: Unsupervised Learning (focus on PCA).

*   **Further Reading:**
    *   **StatQuest with Josh Starmer:**
        *   Principal Component Analysis (PCA): [https://www.youtube.com/watch?v=FgakZw6K1QQ](https://www.youtube.com/watch?v=FgakZw6K1QQ)
        *   t-SNE: [https://www.youtube.com/watch?v=NEaUSP4YerM](https://www.youtube.com/watch?v=NEaUSP4YerM)

## Assignments

*   **Assignment 1: PCA for Image Compression**
    *   Choose an image dataset (e.g., from scikit-learn or TensorFlow Datasets).
    *   Perform PCA on the dataset to reduce the number of dimensions.
    *   Reconstruct the images from the reduced dimensions and visualize the results.
    *   Analyze the trade-off between the number of components retained and the quality of the reconstructed images.

*   **Assignment 2: Visualizing High-Dimensional Data**
    *   Choose a high-dimensional dataset (e.g., from the UCI Machine Learning Repository).
    *   Use PCA, t-SNE, and UMAP to visualize the data in 2D.
    *   Compare the resulting visualizations and discuss the strengths and weaknesses of each technique.

*   **Assignment 3: Feature Selection**
    *   Choose a dataset with a large number of features.
    *   Use different feature selection techniques (e.g., filter, wrapper, and embedded methods) to select a subset of the most important features.
    *   Train a model on the selected features and compare its performance to the performance of a model trained on all of the features.