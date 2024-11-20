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
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {kmeans_score:.4f}")

    # Plot KMeans clustering result
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
    plt.title(f"KMeans (n_clusters={n_clusters})\nSilhouette Score: {kmeans_score:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Find best KMeans parameters based on the highest silhouette score
best_kmeans_params = max(kmeans_silhouette_scores, key=lambda x: x[1])
print(f"Best KMeans Parameters: n_clusters={best_kmeans_params[0]}")
print(f"Best KMeans Silhouette Score: {best_kmeans_params[1]:.4f}")

# Convert to DataFrame for better visualization
silhouette_df = pd.DataFrame(kmeans_silhouette_scores, columns=['Number of Clusters', 'Silhouette Score'])

# Print the DataFrame of silhouette scores
print("\nSilhouette Scores for Each Cluster:")
print(silhouette_df.to_string(index=False))

# Make best KMeans clustering result
best_kmeans = KMeans(n_clusters=best_kmeans_params[0], random_state=42)
best_kmeans_clusters = best_kmeans.fit_predict(X_pca)

# Plot silhouette scores
plt.figure(figsize=(10, 7))
plt.plot(silhouette_df['Number of Clusters'], silhouette_df['Silhouette Score'], marker='o', linestyle='-', color='b')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Plot both best KMeans clustering results
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_kmeans_clusters, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.title(f"Best KMeans (n_clusters={best_kmeans_params[0]})\nSilhouette Score: {best_kmeans_params[1]:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
