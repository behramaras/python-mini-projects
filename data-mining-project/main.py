import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# DATA LOADING AND CLEANING
# ==============================================

# Load data from Excel file
df = pd.read_excel("dataset for mendeley 181220.xlsx")

# Clean unnecessary spaces and quotation marks from column names
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Define columns to be used in the analysis
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

# Select required columns and drop missing values
df = df[required_columns].dropna()

# Identify exam columns (columns containing Math, Science, English)
exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]

# Calculate the average of all exams
df['ExamAverage'] = df[exam_cols].mean(axis=1)
warnings.filterwarnings('ignore')

# ==============================================
# DATA LOADING AND CLEANING
# ==============================================

# Load data from Excel file
df = pd.read_excel("dataset for mendeley 181220.xlsx")

# Clean unnecessary spaces and quotation marks from column names
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Define columns to be used in the analysis
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

# Select required columns and drop missing values
df = df[required_columns].dropna()

# Identify exam columns (columns containing Math, Science, English)
exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]

# Calculate the average of all exams
df['ExamAverage'] = df[exam_cols].mean(axis=1)

# Define function to determine placement level based on exam average
def get_placement_level(avg):
    """
    Determines placement level based on exam average
    85+ : High
    75-84: Medium  
    Below 75: Low
    """
    if avg >= 85:
        return 'High'
    elif avg >= 75:
        return 'Medium'
    else:
        return 'Low'

# Calculate placement level for each student
df['PlacementLevel'] = df['ExamAverage'].apply(get_placement_level)

# ==============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================

# Visualize distribution of placement levels
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='PlacementLevel', order=['Low', 'Medium', 'High'], palette='Set2')
plt.title("PlacementLevel Class Distribution")
plt.tight_layout()
plt.show()

# Plot histogram of exam averages
plt.figure(figsize=(6, 4))
sns.histplot(df['ExamAverage'], kde=True, bins=20, color='skyblue')
plt.title("ExamAverage Distribution")
plt.tight_layout()
plt.show()

# Compare exam averages by gender
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Gender', y='ExamAverage', palette='pastel')
plt.title("Average Scores by Gender")
plt.tight_layout()
plt.show()

# Compare exam averages by previous curriculum
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Previous Curriculum (17/18)2', y='ExamAverage', palette='muted')
plt.title("Average Scores by Curriculum")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Examine correlation between numerical variables
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

# ==============================================
# MODELING – RANDOM FOREST
# ==============================================

# Separate independent variables (X) and target variable (y)
X = df.drop(['ExamAverage', 'PlacementLevel'], axis=1)  # Remove ExamAverage and PlacementLevel
y = df['PlacementLevel']  # Target variable

# Convert categorical variables to one-hot encoded variables
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

# Split data into training (70%), validation (10%), and test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# Get class names (High, Low, Medium)
class_names = np.unique(y)

# ===============================
# FIND BEST RFC PARAMETERS (GRID SEARCH)
# ===============================

# Define parameter combinations to test for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [5, 10, 15],              # Maximum depth
    'min_samples_split': [2, 5],           # Minimum samples required to split
    'min_samples_leaf': [1, 2],            # Minimum samples per leaf
    'max_features': ['sqrt', 'log2'],       # Number of features to consider at each split
    'criterion': ['gini', 'entropy']       # Splitting criterion
}

# Find best parameters using Grid Search
print("Grid Search is starting... This may take some time.")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(bootstrap=True, random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='f1_macro',  # Evaluate based on F1 score
    cv=3,               # 3-fold cross validation
    verbose=2,          # Show progress
    n_jobs=-1           # Use all CPU cores
)

# Run Grid Search
grid_search.fit(X_train, y_train)

print("\nBest parameters (from GridSearchCV):")
best_params = grid_search.best_params_
print(best_params)

# ===============================
# TRAINING AND TESTING WITH BEST MODEL
# ===============================

# Retrieve model with best parameters
best_model = grid_search.best_estimator_

# Train model using training data
print("Training best model...")
best_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = best_model.predict(X_test)              # Class predictions
y_prob = best_model.predict_proba(X_test)        # Probability predictions

# Convert test labels to binary format (required for ROC curve)
y_test_bin = label_binarize(y_test, classes=class_names)

# Make predictions on training data (for overfitting check)
y_train_pred = best_model.predict(X_train)
y_train_prob = best_model.predict_proba(X_train)
y_train_bin = label_binarize(y_train, classes=class_names)

# Calculate basic performance metrics
acc = accuracy_score(y_test, y_pred)                              # Accuracy
prec = precision_score(y_test, y_pred, average='macro')           # Precision
recall = recall_score(y_test, y_pred, average='macro')            # Recall (Sensitivity)
f1 = f1_score(y_test, y_pred, average='macro')                    # F1 Score
auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')  # AUC

# Calculate specificity for each class
specificities = []
for label in class_names:
    # Convert to binary classification for each class
    binary_y_test = (y_test == label).astype(int)
    binary_y_pred = (y_pred == label).astype(int)
    
    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(binary_y_test, binary_y_pred).ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    specificities.append(specificity)

# Calculate macro-average specificity
specificity_macro = np.mean(specificities)

# Print results
print("\nBest model test results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Specificity: {specificity_macro:.4f}")

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.title("Best Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Plot ROC Curves (for both training and test)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # ROC for test data
    fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    # ROC for training data (overfitting check)
    fpr_train, tpr_train, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
    
    # Plot training ROC curve
    axs[0].plot(fpr_train, tpr_train, 
               label=f'{class_label} (Train AUC={roc_auc_score(y_train_bin[:, i], y_train_prob[:, i]):.2f})')
    
    # Plot test ROC curve
    axs[1].plot(fpr_test, tpr_test, 
               label=f'{class_label} (Test AUC={roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.2f})')

# Add random prediction reference line
axs[0].plot([0, 1], [0, 1], 'k--')
axs[1].plot([0, 1], [0, 1], 'k--')

# Set plot titles and labels
axs[0].set_title("Train ROC Curve (Best RFC)")
axs[1].set_title("Test ROC Curve (Best RFC)")

for ax in axs:
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

plt.tight_layout()
plt.show()

# ==============================================
# PERFORMANCE METRICS VISUALIZATION
# ==============================================

# Store all performance metrics in a dictionary
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall (Sensitivity)": recall,
    "F1-Score": f1,
    "AUC": auc,
    "Specificity": specificity_macro
}

# Visualize performance metrics as bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.ylim(0, 1.05)  # Set y-axis limits between 0 and 1
plt.ylabel("Score")
plt.title("Model Performance Metrics (Test Data)")

# Display values on bars
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ==============================================
# FEATURE IMPORTANCE VISUALIZATION
# ==============================================

# Get feature importance scores from Random Forest model
importances = best_model.feature_importances_
feature_names = X.columns

# Convert feature importance to DataFrame and sort
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Visualize most important features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(50), x='Importance', y='Feature', palette='mako')
plt.title("Feature Importance Rankings (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nAnalysis completed! The model was successfully trained and evaluated.")
print(f"Highest performance metric:")
print(f"- Highest score: {max(metrics.values()):.4f}")
print(f"- Corresponding metric: {max(metrics, key=metrics.get)}")
