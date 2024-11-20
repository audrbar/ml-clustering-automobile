from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from data_prep import X_pca
import numpy as np
import matplotlib.pyplot as plt

# Define parameter ranges
dbscan_eps_values = [0.05, 0.1, 0.2]
dbscan_min_samples_values = [10, 15, 20]
dbscan_score = -1
# Initialize lists to store silhouette scores
dbscan_silhouette_scores = []
filtered_dbscan_silhouette_scores = []

# Explore DBSCAN parameters and calculate silhouette scores
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(X_pca)

        # Calculate the percentage of noise points (-1 labels)
        noise_ratio = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

        # Exclude results with more than 50% noise points
        if noise_ratio < 0.5:
            # Filter out noise points for silhouette score
            dbscan_mask = dbscan_clusters != -1
            dbscan_filtered_X_pca = X_pca[dbscan_mask]
            dbscan_filtered_labels = dbscan_clusters[dbscan_mask]

            # Calculate silhouette score if there are clusters formed
            if len(np.unique(dbscan_filtered_labels)) > 1:
                dbscan_score = silhouette_score(dbscan_filtered_X_pca, dbscan_filtered_labels)
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))
            else:
                dbscan_score = np.nan
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))

        # Plot DBSCAN clustering result
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
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
    best_dbscan_clusters = best_dbscan.fit_predict(X_pca)
else:
    print("No suitable DBSCAN parameters found with <50% noise.")
    best_dbscan_params = None

# Plot both best DBSCAN clustering results side-by-side
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_dbscan_clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.title(f"Best DBSCAN (eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]})\nSilhouette Score: "
          f"{best_dbscan_params[2]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
