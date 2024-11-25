import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from data_prep import X, X_pca, X_hist, y_true
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X_hist, y_true, test_size=0.2, random_state=42,
                                                    stratify=y_true)

param_grid = {
    'n_estimators': [32, 34, 36],
    'max_depth': [24, 26, 28],
    'bootstrap': [True]
}

# Parameters to choose from
# 'min_samples_split': [2, 5, 10],
# 'min_samples_leaf': [1, 2, 4],
# 'max_features': ['sqrt', 'log2'],
# 'criterion': ['gini', 'entropy']

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV (Cross Validation)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=6, n_jobs=-1, verbose=2, scoring='accuracy',
                           return_train_score=True)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found:")
print(grid_search.best_params_)

# Get the best estimator from the grid search
best_rf = grid_search.best_estimator_

# Get the cross-validation results
cv_results = grid_search.cv_results_

# ------------------ Evaluate on Train Data ------------------

# Predict on the training set with the best model
y_train_predict = best_rf.predict(X_train)

# Calculate metrics for the training set
train_accuracy = accuracy_score(y_train, y_train_predict)
train_confusion = confusion_matrix(y_train, y_train_predict)
train_precision = precision_score(y_train, y_train_predict, average='macro')
train_recall = recall_score(y_train, y_train_predict, average='macro')
train_f1 = f1_score(y_train, y_train_predict, average='macro')

# Print training set metrics
print("\n--- Training Set Evaluation ---")
print(f"Train Confusion Matrix: \n{train_confusion}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Train Precision: {train_precision:.4f}")
print(f"Train Recall: {train_recall:.4f}")
print(f"Train F1 Score: {train_f1:.4f}")

# ------------------ Evaluate on Test Data ------------------

# Predict on the test set with the best model
y_test_predict = best_rf.predict(X_test)

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_predict)
test_confusion = confusion_matrix(y_test, y_test_predict)
test_precision = precision_score(y_test, y_test_predict, average='macro')
test_recall = recall_score(y_test, y_test_predict, average='macro')
test_f1 = f1_score(y_test, y_test_predict, average='macro')

# Print test set metrics
print("\n--- Test Set Evaluation ---")
print(f"Test Confusion Matrix: \n{test_confusion}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Print each feature along with its importance score
X_df = pd.DataFrame(X_hist)
print("\nFeature Importance:")
for feature, importance in zip(X_df.columns, best_rf.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Extract the mean train and test (validation) scores for each estimator
train_scores = cv_results['mean_train_score']
test_scores = cv_results['mean_test_score']
estimators = [f"n_e: {params['n_estimators']}, m_d: {params['max_depth']}" for params in cv_results['params']]

# Plot the train and test accuracies for each estimator
plt.figure(figsize=(10, 6))
plt.plot(estimators, test_scores, marker='o', linestyle='-', color='g', label='Test Accuracy')
plt.plot(estimators, train_scores, marker='o', linestyle='-', color='b', label='Train Accuracy')
plt.title('Random Forests Overfitting Check: Train and Test Accuracies for Each Estimator')
plt.xlabel('Estimator (n_estimators, max_depth)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame for feature importance's
feature_importance_df = pd.DataFrame({
    'Feature': X_df.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue', edgecolor='k')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.title("Feature Importance's from Random Forest", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Annotate bars with values
for i, value in enumerate(feature_importance_df['Importance']):
    plt.text(i, value + 0.01, f"{value:.2f}", ha='center', fontsize=10)

plt.show()
