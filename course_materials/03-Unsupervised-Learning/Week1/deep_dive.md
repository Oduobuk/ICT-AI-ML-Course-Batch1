# Month 3, Week 1: Clustering - Deep Dive

## K-Means Algorithm Details

*   **Iterative Process:** K-Means is an iterative algorithm that consists of two steps:
    1.  **Assignment Step:** Assign each data point to the cluster with the nearest centroid.
    2.  **Update Step:** Recalculate the centroids of the clusters.
*   **Centroid Initialization:** The initial placement of the centroids can have a significant impact on the final results. K-Means++ is a smart centroid initialization technique that is designed to improve the quality of the clustering.
*   **Convergence Criteria:** The algorithm converges when the centroids no longer move, or when the change in the sum of squared errors (SSE) is below a certain threshold.
*   **Objective Function:** The objective of K-Means is to minimize the SSE, which is the sum of the squared distances between each data point and its assigned centroid.

## Hierarchical Clustering Linkage Methods

*   **Hierarchical Clustering:** A clustering algorithm that builds a hierarchy of clusters. It can be either agglomerative (bottom-up) or divisive (top-down).
*   **Linkage Criteria:** The linkage criterion determines how the distance between two clusters is calculated.
    *   **Single Linkage:** The distance between two clusters is the minimum distance between any two points in the two clusters.
    *   **Complete Linkage:** The distance between two clusters is the maximum distance between any two points in the two clusters.
    *   **Average Linkage:** The distance between two clusters is the average distance between all pairs of points in the two clusters.
    *   **Ward's Linkage:** The distance between two clusters is the increase in the sum of squared errors that results from merging the two clusters.

## DBSCAN Parameters and Strengths

*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** A density-based clustering algorithm that is used to find clusters of arbitrary shape.
*   **Core Parameters:**
    *   **`eps`:** The maximum distance between two points for them to be considered as neighbors.
    *   **`min_samples`:** The minimum number of points required to form a dense region.
*   **Strengths:**
    *   Can find clusters of arbitrary shape.
    *   Can handle noise and outliers.
    *   Does not require the number of clusters to be specified beforehand.

## Clustering Evaluation Metrics

*   **Internal Metrics:** Used to evaluate the quality of the clustering without using any external information.
    *   **Elbow Method:** A method for finding the optimal number of clusters by plotting the SSE as a a function of the number of clusters.
    *   **Silhouette Score:** A measure of how similar a data point is to its own cluster compared to other clusters.
    *   **Davies-Bouldin Index:** A measure of the ratio of within-cluster scatter to between-cluster separation.
    *   **Calinski-Harabasz Index:** A measure of the ratio of between-cluster variance to within-cluster variance.
*   **External Metrics:** Used to evaluate the quality of the clustering using external information, such as ground truth labels.
    *   **Adjusted Rand Index:** A measure of the similarity between two clusterings.
    *   **Mutual Information:** A measure of the mutual dependence between two clusterings.