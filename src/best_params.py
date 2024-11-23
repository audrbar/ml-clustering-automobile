from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from data_prep import X, X_pca, X_pre, X_hist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('dark_background')
plt.style.use('bmh')


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


# --------------------- Agglomerative Clustering ---------------------
agglomerative = AgglomerativeClustering(n_clusters=4, linkage='ward')
agglomerative_labels = agglomerative.fit_predict(X)

# Evaluate Agglomerative Clustering
agglomerative_silhouette = silhouette_score(X, agglomerative_labels)
agglomerative_davies_bouldin = davies_bouldin_score(X, agglomerative_labels)
agglomerative_dunn = calculate_dunn_index(X, agglomerative_labels)

# --------------------- DBSCAN Clustering ---------------------
dbscan = DBSCAN(eps=1.9, min_samples=50)
dbscan_labels = dbscan.fit_predict(X)

# Filter DBSCAN noise for evaluation
non_noise_mask = dbscan_labels != -1
X_non_noise = X[non_noise_mask]
dbscan_filtered_labels = dbscan_labels[non_noise_mask]

# Print information about the filtered data
print(f"\nResults of all Clustering Methods with best parameters:")
print(f"Original dataset size {X.shape[0]} rows reduced to {X_non_noise.shape[0]} non-noise rows.")

if len(np.unique(dbscan_filtered_labels)) > 1:
    dbscan_silhouette = silhouette_score(X_non_noise, dbscan_filtered_labels)
    dbscan_davies_bouldin = davies_bouldin_score(X_non_noise, dbscan_filtered_labels)
    dbscan_dunn = calculate_dunn_index(X_non_noise, dbscan_filtered_labels)
else:
    dbscan_silhouette = np.nan
    dbscan_davies_bouldin = np.nan
    dbscan_dunn = np.nan

# --------------------- K-Means Clustering ---------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_non_noise)

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X_non_noise, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_non_noise, kmeans_labels)
kmeans_dunn = calculate_dunn_index(X_non_noise, kmeans_labels)

# Apply PCA to Reduce Dimensions for Visualization
pca = PCA(n_components=2)
X_pca_non_noise = pca.fit_transform(X_non_noise)

# --------------------- Summary of Clustering Metrics ---------------------
clustering_summary = {
    "Algorithm": ["Agglomerative", "K-Means", "DBSCAN"],
    "Silhouette Score": [f"{agglomerative_silhouette:.4f}", f"{kmeans_silhouette:.4f}", f"{dbscan_silhouette:.4f}"],
    "Davies-Bouldin Score": [f"{agglomerative_davies_bouldin:.4f}", f"{kmeans_davies_bouldin:.4f}",
                             f"{dbscan_davies_bouldin:.4f}"],
    "Dunn Index": [f"{agglomerative_dunn:.4f}", f"{kmeans_dunn:.4f}", f"{dbscan_dunn:.4f}"]
}

clustering_summary_df = pd.DataFrame(clustering_summary)
print("\nClustering Summary:")
print(clustering_summary_df)

# --------------------- Visualization ---------------------
# Plot PCA-reduced Clustering Results
plt.figure(figsize=(18, 8), facecolor='gainsboro')

# Agglomerative
plt.subplot(1, 3, 1)
for cluster in np.unique(agglomerative_labels):
    mask = agglomerative_labels == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}", edgecolor='k', s=50, alpha=0.6)
plt.title(f'Agglomerative Clustering\nSilhouette: {agglomerative_silhouette:.4f}, '
          f'Davies-Bouldin: {agglomerative_davies_bouldin:.4f},\nDunn Index: {agglomerative_dunn:.4f}')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# K-Means
plt.subplot(1, 3, 2)
for cluster in np.unique(kmeans_labels):
    mask = kmeans_labels == cluster
    plt.scatter(X_pca_non_noise[mask, 0], X_pca_non_noise[mask, 1], label=f"Cluster {cluster}", edgecolor='k',
                s=50, alpha=0.6)
plt.title(f'K-Means Clustering\nSilhouette: {kmeans_silhouette:.4f}, Davies-Bouldin: {kmeans_davies_bouldin:.4f} ,'
          f'\nDunn Index: {kmeans_dunn:.4f}')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# DBSCAN
plt.subplot(1, 3, 3)
for cluster in np.unique(dbscan_labels):
    mask = dbscan_labels == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster}" if cluster != -1 else "Noise",
                edgecolor='k', s=50, alpha=0.6)
plt.title(f'DBSCAN Clustering\nSilhouette: {dbscan_silhouette:.4f}, Davies-Bouldin: {dbscan_davies_bouldin:.4f}, '
          f'\nDunn Index: {dbscan_dunn:.4f}')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()

# --------------------- Features Distribution Across Clusters on KMeans Clustering ---------------------
# Apply DBSCAN clustering
dbscan_labels_ = dbscan.fit_predict(X)

# Filter DBSCAN noise for evaluation
non_noise_mask_ = dbscan_labels_ != -1
X_non_noise_ = X_hist[non_noise_mask_]
dbscan_filtered_labels_ = dbscan_labels_[non_noise_mask_]

# Apply K-Means clustering
kmeans_labels_ = kmeans.fit_predict(X_non_noise_)

# Number of features in X and number of unique clusters
n_features = X_non_noise_.shape[1]
feature_names = ['Gender', 'Married', 'Age', 'Graduated', 'Profession', 'WorkExperience',
                 'SpendingScore', 'FamilySize', 'Category']
clusters_ = np.unique(kmeans_labels_)

# Define appropriate xticks for each feature based on their categorical mapping
xtick_labels = {
    'Gender': ['Male', 'Female'],
    'Married': ['No', 'Yes'],
    'Graduated': ['No', 'Yes'],
    'Profession': ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'],
    'SpendingScore': ['Low', 'Average', 'High'],
    'FamilySize': ['Small', 'Medium', 'Large'],
    'Category': ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7']
}

# Convert X_non_noise_ to a NumPy array if it's a DataFrame
if isinstance(X_non_noise_, pd.DataFrame):
    X_non_noise_ = X_non_noise_.values

# Plot histograms for each cluster
plt.figure(figsize=(18, 8), facecolor='gainsboro')

for i, feature in enumerate(feature_names):
    plt.subplot(3, 3, i + 1)
    for cluster in np.unique(kmeans_labels_):
        cluster_data_ = X_non_noise_[kmeans_labels_ == cluster, i]
        plt.hist(cluster_data_, bins=np.arange(cluster_data_.min() - 0.5, cluster_data_.max() + 1.5), alpha=0.5,
                 edgecolor='k', label=f'Cluster {cluster}')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    # Add appropriate xticks for categorical features
    if feature in xtick_labels and feature not in ['Age', 'WorkExperience']:
        plt.xticks(ticks=range(len(xtick_labels[feature])), labels=xtick_labels[feature], rotation=45)
    else:
        plt.xticks(rotation=45)

    plt.legend()

plt.tight_layout()
plt.suptitle("Features Distribution Across Clusters", fontsize=12)
plt.subplots_adjust(top=0.92)
plt.show()

# ---------------------- Customer Profiles for Each Cluster --------------------------

# Convert X_non_noise_ back to DataFrame for easier profiling
X_non_noise_df = pd.DataFrame(X_non_noise_, columns=feature_names)

# Add cluster labels to the DataFrame
X_non_noise_df['Cluster'] = kmeans_labels_


# Function to compute profiles for each cluster
def compute_cluster_profiles(df, feature_names_):
    profiles = []
    clusters = df['Cluster'].unique()

    for cluster_ in clusters:
        cluster_data = df[df['Cluster'] == cluster_]
        profile = {'Cluster': cluster_}

        # Compute mean for numeric features
        for feature_ in feature_names_:
            if feature_ not in ['Age', 'WorkExperience']:
                profile[feature_] = cluster_data[feature_].mean()
            else:
                # Compute the most common category for Age and WorkExperience
                profile[feature] = cluster_data[feature_].mode()[0]

        profiles.append(profile)

    return pd.DataFrame(profiles)


# Compute cluster profiles
cluster_profiles = compute_cluster_profiles(X_non_noise_df, feature_names).sort_values(by='Cluster')

# Print profiles for each cluster
print("\nCustomer Profiles for Each Cluster:")
print(cluster_profiles)

# Plot sorted cluster profiles with bar values
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the bar chart
cluster_profiles.set_index('Cluster').plot(kind='bar', ax=ax, legend=True)
plt.title('Sorted Customer Profiles for Each Cluster')
plt.ylabel('Normalized Feature Value')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.tight_layout()

# Add values on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=9, padding=3)

plt.show()
