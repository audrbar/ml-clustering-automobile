from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
data = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering-automobile/data/train-set.csv')

# Clean and Prepare Data
df_cleaned = data.drop(columns=['CustomerID']).dropna()  # Drop CustomerID and missing values

# Encode Categorical Variables
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

# Scale Numerical Features
scaler = StandardScaler()
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])

# Prepare Data for Clustering
X = df_cleaned.drop(columns=['Segmentation']).values  # Features
y_true = df_cleaned['Segmentation'].values  # True labels (if available for evaluation)

# Apply PCA to Reduce Dimensions for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --------------------- K-Means Clustering ---------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)

# --------------------- DBSCAN Clustering ---------------------
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)

# Filter DBSCAN noise for evaluation
dbscan_mask = dbscan_labels != -1
dbscan_filtered_data = X[dbscan_mask]
dbscan_filtered_labels = dbscan_labels[dbscan_mask]

if len(np.unique(dbscan_filtered_labels)) > 1:
    dbscan_silhouette = silhouette_score(dbscan_filtered_data, dbscan_filtered_labels)
    dbscan_davies_bouldin = davies_bouldin_score(dbscan_filtered_data, dbscan_filtered_labels)
else:
    dbscan_silhouette = np.nan
    dbscan_davies_bouldin = np.nan

# --------------------- Agglomerative Clustering ---------------------
agglo = AgglomerativeClustering(n_clusters=4, linkage='complete')
agglo_labels = agglo.fit_predict(X)

# Evaluate Agglomerative Clustering
agglo_silhouette = silhouette_score(X, agglo_labels)
agglo_davies_bouldin = davies_bouldin_score(X, agglo_labels)

# --------------------- Visualization ---------------------
# Plot PCA-reduced Clustering Results
plt.figure(figsize=(18, 7))

# K-Means
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f'K-Means Clustering\nSilhouette: {kmeans_silhouette:.2f}, Davies-Bouldin: {kmeans_davies_bouldin:.2f}')

# DBSCAN
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f'DBSCAN Clustering\nSilhouette: {dbscan_silhouette:.2f}, Davies-Bouldin: {dbscan_davies_bouldin:.2f}')

# Agglomerative
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agglo_labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f'Agglomerative Clustering\nSilhouette: {agglo_silhouette:.2f}, Davies-Bouldin: {agglo_davies_bouldin:.2f}')

plt.tight_layout()
plt.show()

# --------------------- Dendrogram for Agglomerative Clustering ---------------------
plt.figure(figsize=(10, 7))
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# --------------------- Summary of Clustering Metrics ---------------------
clustering_summary = {
    "Algorithm": ["K-Means", "DBSCAN", "Agglomerative"],
    "Silhouette Score": [kmeans_silhouette, dbscan_silhouette, agglo_silhouette],
    "Davies-Bouldin Score": [kmeans_davies_bouldin, dbscan_davies_bouldin, agglo_davies_bouldin]
}

clustering_summary_df = pd.DataFrame(clustering_summary)
print("\nClustering Summary:")
print(clustering_summary_df)
