# Month 3, Week 2: Dimensionality Reduction - Deep Dive

## PCA Mathematical Derivation

*   **Eigenvectors and Eigenvalues:** An eigenvector of a square matrix is a non-zero vector that, when multiplied by the matrix, is simply scaled by a scalar factor called the eigenvalue. In the context of PCA, the eigenvectors of the covariance matrix of the data are the principal components, and the eigenvalues are the variance of the data along the principal components.
*   **Covariance Matrix:** A square matrix that measures the covariance between each pair of features in the data. The diagonal elements of the covariance matrix are the variances of each feature.
*   **Finding the Principal Components:** The principal components are the eigenvectors of the covariance matrix of the data. The first principal component is the eigenvector with the largest eigenvalue, the second principal component is the eigenvector with the second largest eigenvalue, and so on.

## Scree Plots and Explained Variance Ratio

*   **Scree Plot:** A plot of the eigenvalues of the covariance matrix in descending order. It can be used to determine the optimal number of principal components to retain.
*   **Explained Variance Ratio:** The percentage of the total variance that is explained by each principal component. It can be used to determine the optimal number of principal components to retain.

## t-SNE and UMAP Algorithms

*   **t-SNE (t-distributed Stochastic Neighbor Embedding):** A non-linear dimensionality reduction technique that is used for visualizing high-dimensional data. It works by converting the high-dimensional data into a low-dimensional representation that preserves the local structure of the data.
*   **UMAP (Uniform Manifold Approximation and Projection):** A non-linear dimensionality reduction technique that is similar to t-SNE. It is often faster than t-SNE and can produce better visualizations.

## Manifold Learning

*   **Manifold:** A topological space that is locally Euclidean. In the context of machine learning, a manifold is a lower-dimensional surface on which the high-dimensional data lies.
*   **Manifold Learning:** A class of algorithms that are used to learn the underlying manifold of the data. t-SNE and UMAP are examples of manifold learning algorithms.

## Feature Selection Techniques

*   **Filter Methods:** Select features based on their statistical properties, such as their correlation with the target variable or their mutual information.
*   **Wrapper Methods:** Select features by training a model on different subsets of the features and then selecting the subset of features that results in the best performance.
*   **Embedded Methods:** Select features as part of the model training process. Lasso and tree-based feature importance are examples of embedded methods.