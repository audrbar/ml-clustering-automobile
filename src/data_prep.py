from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# Define a function to categorize age
def categorize_age(age):
    if age < 13:
        return 0  # Child
    elif age < 18:
        return 1  # Teen
    elif age < 35:
        return 2  # Young Adult
    elif age < 50:
        return 3  # Adult
    elif age < 65:
        return 4  # Middle Age
    else:
        return 5  # Senior


# Load Data
pd.options.display.max_columns = None
initial_df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering-automobile/data/train-set.csv')

# Display dataset information for exploration
print("\nInitial Dataset Info:")
print(initial_df.info())

# Display unique values for each column
print("\nInitial Dataset Unique Values for Each Column:")
for col in initial_df.columns:
    print(f"{col}: {initial_df[col].unique()}")

# Clean and Prepare Data
df = initial_df.drop(columns=['CustomerID', 'Married']).dropna()  # Or: df = df.fillna(df.mean())

# Apply the function to the Age column
df['Age'] = df['Age'].apply(categorize_age)
df['Segmentation'] = df['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"\nCategorical Columns: \n{categorical_columns}")
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Scale Numerical Features
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0, 1))
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
filtered_columns = numeric_columns.columns[numeric_columns.max() > 12]
df[filtered_columns] = scaler.fit_transform(df[filtered_columns])
print(f"\nNumeric Columns: \n{numeric_columns.columns}")
print(f"\nFiltered Columns: \n{filtered_columns}")

# Final inspection of the preprocessed dataset
print("\nCleaned Dataset Info:")
print(df.info())
print("\nCleaned Dataset Unique Values for Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# Prepare Data for Clustering
X = df.drop(columns=['Segmentation']).values  # Features
y_true = df['Segmentation'].values  # True labels (if available for evaluation)

# Apply PCA to Reduce Dimensions for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Final preprocessed data for analysis
print("\nFinal preprocessed X:")
print(X)
print("\nFinal preprocessed y_true:")
print(y_true)
print("\nFinal preprocessed X_pca:")
print(X_pca)

# # Calculate the correlation matrix of the Initial Dataset
# correlation_matrix_df = df.corr()
# #
# # Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap of the Initial Dataset')
# plt.show()
