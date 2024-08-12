import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder # type: ignore
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Step 1: Handle missing values
# Numerical features - Replace missing values with the mean of the column.
# Categorical features - Replace missing values with the most frequent value.
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(exclude=[np.number]).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Replacing missing values with the mean of the column
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replacing missing values with the most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding for categorical features
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 2: Standardization/Normalization
# StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()

# Step 3: Dimensionality Reduction
# PCA is used to reduce the number of features while retaining as much variance as possible.
pca = PCA(n_components=0.95)  # Retain 95% of the variance

# Step 4: Pipeline Integration
# The full pipeline integrates the preprocessing steps with a machine learning model.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', scaler),
    ('pca', pca),
    ('classifier', RandomForestClassifier())  # Example classifier (you can choose your model)
])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Fit the model
model.fit(X_train, y_train)

# Step 7: Evaluate the model
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

