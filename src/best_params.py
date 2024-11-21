from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from data_prep import X, X_pca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_dunn_index(_X, labels):
    """Calculate the Dunn Index for a clustering result."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan  # Cannot calculate Dunn Index for single or no clusters

    # Calculate inter-cluster distances (min distances between clusters)
    inter_cluster_distances = []
    for i in unique_labels:
        for j in unique_labels:
            if i != j:
                cluster_i = _X[labels == i]
                cluster_j = _X[labels == j]
                inter_dist = np.min(pairwise_distances(cluster_i, cluster_j))
                inter_cluster_distances.append(inter_dist)

    # Calculate intra-cluster distances (max distance within a cluster)
    intra_cluster_distances = []
    for i in unique_labels:
        cluster_i = _X[labels == i]
        intra_dist = np.max(pairwise_distances(cluster_i))
        intra_cluster_distances.append(intra_dist)

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)


# --------------------- K-Means Clustering ---------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)
kmeans_dunn = calculate_dunn_index(X, kmeans_labels)

# --------------------- DBSCAN Clustering ---------------------
dbscan = DBSCAN(eps=0.2, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)

# Filter DBSCAN noise for evaluation
dbscan_mask = dbscan_labels != -1
dbscan_filtered_data = X[dbscan_mask]
dbscan_filtered_labels = dbscan_labels[dbscan_mask]

if len(np.unique(dbscan_filtered_labels)) > 1:
    dbscan_silhouette = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
    dbscan_davies_bouldin = davies_bouldin_score(dbscan_filtered_data, dbscan_filtered_labels)
    dbscan_dunn = calculate_dunn_index(dbscan_filtered_data, dbscan_filtered_labels)
else:
    dbscan_silhouette = np.nan
    dbscan_davies_bouldin = np.nan
    dbscan_dunn = np.nan

# --------------------- Agglomerative Clustering ---------------------
agglomerative = AgglomerativeClustering(n_clusters=4, linkage='complete')
agglomerative_labels = agglomerative.fit_predict(X)

# Evaluate Agglomerative Clustering
agglomerative_silhouette = silhouette_score(X, agglomerative_labels)
agglomerative_davies_bouldin = davies_bouldin_score(X, agglomerative_labels)
agglomerative_dunn = calculate_dunn_index(X, agglomerative_labels)

# --------------------- Summary of Clustering Metrics ---------------------
clustering_summary = {
    "Algorithm": ["K-Means", "DBSCAN", "Agglomerative"],
    "Silhouette Score": [f"{kmeans_silhouette:.4f}", f"{dbscan_silhouette:.4f}", f"{agglomerative_silhouette:.4f}"],
    "Davies-Bouldin Score": [f"{kmeans_davies_bouldin:.4f}", f"{dbscan_davies_bouldin:.4f}",
                             f"{agglomerative_davies_bouldin:.4f}"],
    "Dunn Index": [f"{kmeans_dunn:.4f}", f"{dbscan_dunn:.4f}", f"{agglomerative_dunn:.4f}"]
}

clustering_summary_df = pd.DataFrame(clustering_summary)
print("\nClustering Summary:")
print(clustering_summary_df)

# --------------------- Visualization ---------------------
# Plot PCA-reduced Clustering Results
plt.figure(figsize=(18, 7))

# K-Means
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50, alpha=0.6)
plt.title(f'K-Means Clustering\nSilhouette: {kmeans_silhouette:.2f}, Davies-Bouldin: {kmeans_davies_bouldin:.2f}')

# DBSCAN
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', edgecolor='k', s=50, alpha=0.6)
plt.title(f'DBSCAN Clustering\nSilhouette: {dbscan_silhouette:.2f}, Davies-Bouldin: {dbscan_davies_bouldin:.2f}')

# Agglomerative
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglomerative_labels, cmap='viridis', edgecolor='k', s=50, alpha=0.6)
plt.title(f'Agglomerative Clustering\nSilhouette: {agglomerative_silhouette:.2f}, '
          f'Davies-Bouldin: {agglomerative_davies_bouldin:.2f}')

plt.tight_layout()
plt.show()
