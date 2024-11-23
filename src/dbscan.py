from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from data_prep import X, X_pca
import numpy as np
import matplotlib.pyplot as plt

# Define parameter ranges
dbscan_eps_values = [1.8, 1.9, 2]
dbscan_min_samples_values = [20, 30, 50]
dbscan_score = -1

# Initialize lists to store silhouette scores
dbscan_silhouette_scores = []
filtered_dbscan_silhouette_scores = []

# Explore DBSCAN parameters and calculate silhouette scores

print(f"\nResults of the DBSCAN Clustering for each eps and min_samples values:")
for eps in dbscan_eps_values:
    for min_samples in dbscan_min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(X)

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
                print(f"eps: {eps}, min samples: {min_samples}, Silhouette Score: {dbscan_score:.4f}, "
                      f"noise ratio: {noise_ratio:.2%}")
            else:
                dbscan_score = np.nan
                filtered_dbscan_silhouette_scores.append((eps, min_samples, dbscan_score, noise_ratio))

        # Plot DBSCAN clustering result
        plt.figure(figsize=(12, 7))
        for cluster in np.unique(dbscan_clusters):
            mask = dbscan_clusters == cluster
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}" if cluster != -1 else "Noise",
                        edgecolor='k', s=50, alpha=0.6)
        plt.title(f'DBSCAN Clustering\neps: {eps}, min_samples: {min_samples}, Silhouette: {dbscan_score:.4f}')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

# Find the best DBSCAN parameters based on the highest silhouette score (from filtered results)
if filtered_dbscan_silhouette_scores:
    best_dbscan_params = max(filtered_dbscan_silhouette_scores, key=lambda x: x[2])
    print(f"\nBest DBSCAN Parameters:\neps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]}, "
          f"Silhouette Score: {best_dbscan_params[2]:.4f}")

    print(f"Noise Ratio with Best Params: {best_dbscan_params[3]:.2%}")

    # Make best DBSCAN clustering result
    best_dbscan = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
    best_dbscan_clusters = best_dbscan.fit_predict(X_pca)
else:
    print("No suitable DBSCAN parameters found with <50% noise.")
    best_dbscan_params = None
