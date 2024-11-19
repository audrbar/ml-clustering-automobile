from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_prep_ import X
from sklearn.decomposition import PCA
import numpy as np

# Apply PCA to identify the optimal number of components
pca = PCA()
pca.fit(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Select the top 2 principal components for visualization
pca_optimal = PCA(n_components=2)
X_pca = pca_optimal.fit_transform(X)

# Assess the quality of K-Means clusters using Within-Cluster Sum of Squares (WCSS)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=600, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# The optimal number of clusters (k) - Elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('The optimal number of clusters (k) - Elbow method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# Define optimal k (from previous Elbow Method)
optimal_k = 4

# Train KMeans on PCA-reduced data
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=600, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# Print K-Means model details
print("\nK-Means Best Parameters:")
print(f"Number of Clusters: {optimal_k}")
print(f"Inertia (WCSS): {kmeans.inertia_:.4f}")
print(f"Cluster Centers:\n{kmeans.cluster_centers_}")

# Plot the clustering result for PCA-reduced data
plt.figure(figsize=(12, 7))
for cluster in np.unique(y_kmeans):
    plt.scatter(
        X_pca[y_kmeans == cluster, 0],
        X_pca[y_kmeans == cluster, 1],
        label=f'Cluster {cluster}',
        s=50, edgecolor='k'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=150, c='red', label='Centroids', edgecolor='k', marker='X'
)
plt.title(f'K-means Clustering (PCA Data, k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
