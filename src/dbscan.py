from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from data_prep_ import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
data = pca.fit_transform(X)

# Define parameter ranges
dbscan_eps_values = [0.3, 0.4, 0.5]
dbscan_min_samples_values = [3, 5, 10]
kmeans_cluster_values = [3, 5, 7]

# Initialize lists to store silhouette scores
dbscan_silhouette_scores = []
kmeans_silhouette_scores = []
filtered_dbscan_silhouette_scores = []

# Explore DBSCAN parameters and calculate silhouette scores
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(data)

        # Calculate the percentage of noise points (-1 labels)
        noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

        # Exclude results with more than 50% noise points
        if noise_ratio < 0.5:
            # Filter out noise points for silhouette score
            dbscan_mask = dbscan_clusters != -1
            dbscan_filtered_data = data[dbscan_mask]
            dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

            # Calculate silhouette score if there are clusters formed
            if len(np.unique(dbscan_filtered_labels)) > 1:
                dbscan_score = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))
            else:
                dbscan_score = np.nan
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))

        # Plot DBSCAN clustering result
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=dbscan_clusters, cmap='plasma', marker='o', edgecolor='k')
        plt.title(f"DBSCAN (eps={eps}, min_samples={min_samples})\nSilhouette Score: {dbscan_score:.4f}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

# Find the best DBSCAN parameters based on the highest silhouette score (from filtered results)
if filtered_dbscan_silhouette_scores:
    best_dbscan_params = max(filtered_dbscan_silhouette_scores, key=lambda x: x[2])
    print(f"Best DBSCAN Parameters: eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]}")
    print(f"Best DBSCAN Silhouette Score: {best_dbscan_params[2]:.4f}")
    print(f"Noise Ratio with Best Params: {best_dbscan_params[3]:.2%}")

    # Make best DBSCAN clustering result
    best_dbscan = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
    best_dbscan_clusters = best_dbscan.fit_predict(data)
else:
    print("No suitable DBSCAN parameters found with <50% noise.")
    best_dbscan_params = None

# Explore KMeans parameters and calculate silhouette scores
for n_clusters in kmeans_cluster_values:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans.fit_predict(data)

    # Calculate silhouette score for KMeans
    kmeans_score = silhouette_score(data, kmeans_clusters)
    kmeans_silhouette_scores.append((n_clusters, kmeans_score))

    # Plot KMeans clustering result
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_clusters, cmap='plasma', marker='o', edgecolor='k')
    plt.title(f"KMeans (n_clusters={n_clusters})\nSilhouette Score: {kmeans_score:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Find best KMeans parameters based on the highest silhouette score
best_kmeans_params = max(kmeans_silhouette_scores, key=lambda x: x[1])
print(f"Best KMeans Parameters: n_clusters={best_kmeans_params[0]}")
print(f"Best KMeans Silhouette Score: {best_kmeans_params[1]:.4f}")

# Make best KMeans clustering result
best_kmeans = KMeans(n_clusters=best_kmeans_params[0], random_state=42)
best_kmeans_clusters = best_kmeans.fit_predict(data)

# Plot both best DBSCAN and KMeans clustering results side-by-side
plt.figure(figsize=(14, 6))

# DBSCAN subplot
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=best_dbscan_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title(f"Best DBSCAN (eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]})\nSilhouette Score: {best_dbscan_params[2]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# KMeans subplot
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=best_kmeans_clusters, cmap='plasma', marker='o', edgecolor='k')
plt.title(f"Best KMeans (n_clusters={best_kmeans_params[0]})\nSilhouette Score: {best_kmeans_params[1]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
