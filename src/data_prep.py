from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler


# Define a function to categorize age
def categorize_age(age):
    if age < 21:
        return 0  # Youth
    elif age < 35:
        return 1  # Young Adult
    elif age < 50:
        return 2  # Adult
    elif age < 65:
        return 3  # Mature Adult
    else:
        return 4  # Senior


def categorize_experience(experience):
    if experience < 1:
        return 0  # very low
    elif experience < 3:
        return 1  # low
    elif experience < 6:
        return 2  # moderate
    elif experience < 9:
        return 3  # high
    elif experience < 12:
        return 4  # very high
    else:
        return 5  # exceptional


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
df = initial_df.drop(columns=['CustomerID']).dropna()  # Or: df = df.fillna(df.mean())

# Calculate target class balance and add percentage column
target_balance = df['Segmentation'].value_counts().reset_index()
target_balance.columns = ['Segmentation', 'Count']  # Rename columns for clarity
target_balance['Percentage'] = (target_balance['Count'] / target_balance['Count'].sum()) * 100
print("\nTarget Class Balance:")
for index, row in target_balance.iterrows():
    print(f"{row['Segmentation']} - {row['Count']}, {row['Percentage']:.1f}%")

# Apply the functions or mapping to the column to categorize data instead using LabelEncoder
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
df['Graduated'] = df['Graduated'].map({'No': 0, 'Yes': 1})
df['Profession'] = df['Profession'].map({'Healthcare': 0, 'Engineer': 1, 'Lawyer': 2, 'Entertainment': 3,
                                         'Artist': 4, 'Executive': 5, 'Doctor': 6, 'Homemaker': 7, 'Marketing': 8})
df['SpendingScore'] = df['SpendingScore'].map({'Low': 0, 'Average': 1, 'High': 2})
df['Category'] = df['Category'].map({'Category 1': 0, 'Category 2': 1, 'Category 3': 2, 'Category 4': 3,
                                     'Category 5': 4, 'Category 6': 5, 'Category 7': 6})
df['Segmentation'] = df['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
X_hist = df.drop(columns=['Segmentation'])
df['Age'] = df['Age'].apply(categorize_age)
df['WorkExperience'] = df['WorkExperience'].apply(categorize_experience)

# Final inspection of the preprocessed dataset
print("\nCleaned and Preprocessed Dataset Info:")
print(df.info())
print("\nCleaned Dataset Unique Values for Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# Prepare Data for Clustering
X_pre = df.drop(columns=['Segmentation']).values  # Features
y_true = df['Segmentation'].values  # True labels for evaluation

# Apply scaling (StandardScaler) on the X_pre
scaler = StandardScaler()
X = scaler.fit_transform(X_pre)

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

# # Define the number of rows and columns for the subplots
# n_features = len(df.columns)
# n_cols = 5  # Adjust as needed
# n_rows = math.ceil(n_features / n_cols)
#
# # Create the figure and axes
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, n_rows * 4))
# axes = axes.flatten()  # Flatten axes for easier iteration
#
# # Plot each feature in a subplot
# for i, column in enumerate(df.columns):
#     # Get the current axis
#     ax = axes[i]
#
#     # Plot value counts
#     df[column].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
#
#     # Set titles and labels
#     ax.set_title(f'Distribution of {column}', fontsize=12)
#     ax.set_xlabel(column, fontsize=10)
#     ax.set_ylabel('Frequency', fontsize=10)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Adjust layout
# plt.tight_layout()
# plt.show()
#
# # Calculate the correlation matrix of the Initial Dataset
# correlation_matrix_df = df.corr()
# #
# # Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap of the Initial Dataset')
# plt.show()
