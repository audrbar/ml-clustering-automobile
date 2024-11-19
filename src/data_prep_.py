from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Data
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering-automobile/data/cinemaTicket_Ref.csv')

# Handle missing values (if any)
df = df.dropna()  # Or: df = df.fillna(df.mean())

# Drop rows starting from index
df = df.iloc[:5000]

# Drop unnecessary columns (CustomerID, Category)
df = df.drop(df.columns[[10, 12, 13]], axis=1)

# Display dataset information for exploration
print("\nDataset Info:")
print(df.info())

# Display unique values for each column to understand the data better
print("\nUnique Values for Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# Normalize numeric columns where scaling would be beneficial
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
filtered_columns = numeric_columns.loc[:, numeric_columns.max() > 12]
print(f"\nColumns to be Normalized: \n{filtered_columns.columns}")

# Apply StandardScaler only to the filtered columns
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0, 1))
df[filtered_columns.columns] = scaler.fit_transform(df[filtered_columns.columns])

# Final inspection of the preprocessed dataset
print("\nData Summary After Normalization:")
print(df.describe())

# Calculate the correlation matrix for numeric columns
correlation_matrix_df = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap DF')
plt.show()

X = df.drop(columns=['film_code', 'ticket_use', 'tickets_sold'])
y_true = df[['film_code']]
print(y_true)
le = LabelEncoder()
y_true = le.fit_transform(y_true)
print(y_true)

# Calculate the correlation matrix for numeric columns
correlation_matrix_x = X.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(correlation_matrix_x, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap X')
plt.show()
