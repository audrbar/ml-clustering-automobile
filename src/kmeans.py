import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_prep import X_pca
import pandas as pd
import matplotlib.pyplot as plt

# Calculate Within Cluster Sum of Squares (WCSS) for different numbers of clusters
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=600, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# The optimal number of clusters (k) - Elbow method
plt.figure(figsize=(10, 7))
plt.plot(k_range, wcss, marker='o')
plt.title('The optimal number of clusters (k) - Elbow method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

print(f"\nResults of the KMeans Clustering for each number of clusters:")
# Define parameter ranges
kmeans_cluster_values = [2, 3, 4, 5, 6]

# Initialize lists to store silhouette scores
kmeans_silhouette_scores = []

# Explore KMeans parameters and calculate silhouette scores
for n_clusters in kmeans_cluster_values:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(X_pca)

    # Calculate silhouette score for KMeans
    kmeans_score = silhouette_score(X_pca, kmeans_clusters)
    kmeans_silhouette_scores.append((n_clusters, kmeans_score))
    print(f"{n_clusters} clusters Silhouette Score: {kmeans_score:.4f}")

    # Plot KMeans clustering result
    plt.figure(figsize=(10, 7))
    for cluster in np.unique(kmeans_clusters):
        mask = kmeans_clusters == cluster
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}", edgecolor='k', s=50, alpha=0.6)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
    plt.title(f"KMeans Clustering\nNumber of Clusters: {n_clusters}, Silhouette Score: {kmeans_score:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Find best KMeans parameters based on the highest silhouette score
best_kmeans_params = max(kmeans_silhouette_scores, key=lambda x: x[1])
print("\nBest KMeans Clustering Parameters:")
print(f"Number of clusters: {best_kmeans_params[0]}, Silhouette Score: {best_kmeans_params[1]:.4f}")
