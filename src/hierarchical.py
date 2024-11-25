import sys
from data_prep import X, y_true, X_pca
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Increase recursion limit to handle larger dendrograms
sys.setrecursionlimit(4000)

# Calculate Within Cluster Sum of Squares (WCSS) for different numbers of clusters
wcss = []
max_clusters = 8
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='complete')
    labels = agglomerative.fit_predict(X)

    # Calculate WCSS for the clusters formed by Agglomerative Clustering
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Calculate WCSS
    _, dists = pairwise_distances_argmin_min(X, centroids)
    wcss.append(np.sum(dists ** 2))

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering')
plt.xlabel('Number of Clusters (n_clusters)')
plt.ylabel('Within Cluster Sum of Squares (WCSS)')
plt.show()

# Define optimal k (from previous Elbow Method)
optimal_clusters = 4
print(f"\nResults of the Hierarchical (Agglomerative) Clustering for each linkage method:\nOptimal number of clusters: "
      f"{optimal_clusters}")

# List of linkage methods  and accuracy list
linkage_methods = ['ward', 'complete', 'average', 'single']
accuracies = []

# Apply hierarchical clustering and plot results for each linkage method
for method in linkage_methods:
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=method)
    y_predicted = clustering.fit_predict(X)

    # Calculate accuracy if ground truth is available
    try:
        accuracy = accuracy_score(y_true, y_predicted)
        accuracies.append((method, accuracy))
        print(f"'{method.capitalize()} Linkage' accuracy: {accuracy:.4f}")
    except NameError:
        print("Ground truth (y_true) not provided, skipping accuracy calculation.")
        accuracies.append((method, None))

    # Plot dendrogram
    plt.figure(figsize=(12, 7))
    Z = linkage(X, method=method)
    dendrogram(Z)
    plt.title(f'Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

    # Plot clusters in PCA-reduced space
    plt.figure(figsize=(12, 7))
    for cluster in np.unique(y_predicted):
        plt.scatter(
            X_pca[y_predicted == cluster, 0],
            X_pca[y_predicted == cluster, 1],
            label=f'Cluster {cluster}',
            s=50, edgecolor='k', alpha=0.6
        )
    plt.title(
        f"Hierarchical (Agglomerative) Clustering\n'{method.capitalize()} Linkage' accuracy: {accuracy:.4f}",
        fontsize=14
    )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

best_agglomerative_params = max(accuracies, key=lambda x: x[1])
print("\nBest Hierarchical (Agglomerative) Clustering Parameters:")
print(f"Method: {best_agglomerative_params[0]}, Silhouette Score: {best_agglomerative_params[1]:.4f}")
