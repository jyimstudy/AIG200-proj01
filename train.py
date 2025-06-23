import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from category_encoders import TargetEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib

print("Script started: Loading and preprocessing data...")

# --- 1. Data Loading and Initial Cleaning ---
# Load the dataset from the CSV file
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Drop rows with 'unknown' in 'marital' as decided in the notebook
df = df[df['marital'] != 'unknown'].copy()

# Drop 'default' & 'duration' columns. 'duration' is dropped to prevent data leakage.
df.drop('default', axis=1, inplace=True)
df.drop('duration', axis=1, inplace=True)

# Encode target variable 'y' to 0 and 1
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# --- 2. Feature Engineering and Encoding ---
# Binary encode 'contact'
df['contact'] = df['contact'].map({'telephone': 0, 'cellular': 1})

# Binary encode 'housing' and 'loan' using a consistent map
label_map = {'no': 0, 'yes': 1, 'unknown': 2}
df['housing'] = df['housing'].map(label_map)
df['loan'] = df['loan'].map(label_map)

# Ordinal encode 'education' based on assumed hierarchy
edu_order = {
    'illiterate': 0, 'unknown': 1, 'basic.4y': 2, 'basic.6y': 3,
    'basic.9y': 4, 'high.school': 5, 'professional.course': 6, 'university.degree': 7
}
df['education'] = df['education'].map(edu_order)

# Map 'month' to numerical for potential seasonal patterns
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df['month'] = df['month'].map(month_map)

# Create binary features from 'pdays' and 'previous'
df['previously_contacted'] = np.where(df['pdays'] != 999, 1, 0)
df['was_contacted_before'] = np.where(df['previous'] > 0, 1, 0)
df.drop(columns=['pdays', 'previous'], inplace=True)

# Log transform 'campaign' to reduce skewness
df['campaign'] = np.log1p(df['campaign'])

# One-hot encode 'marital' and 'day_of_week'
df = pd.get_dummies(df, columns=['marital', 'day_of_week'], drop_first=True)

print("Preprocessing complete. Splitting data...")

# --- 3. Data Splitting and Final Processing ---
# Separate features (X) and target (y)
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Target encode 'job' and 'poutcome'
target_cols = ['job', 'poutcome']
target_enc = TargetEncoder(cols=target_cols)
X_train[target_cols] = target_enc.fit_transform(X_train[target_cols], y_train)
X_test[target_cols] = target_enc.transform(X_test[target_cols])

# Standardize numerical features
numeric_cols = ['age', 'campaign', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Ensure column order is the same for test set
X_test = X_test[X_train.columns]

print("Data splitting and final processing complete.")

# --- 4. Model Training (Config A) ---
print("Training Config A model...")
# Calculate class weights for handling imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Build the model architecture for Config A
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

print("Model training complete.")

# --- 5. Saving Artifacts ---
print("Saving model, encoders, and scaler...")

# Save the trained Keras model
model.save('bank_marketing_model.keras')

# Save the fitted TargetEncoder
joblib.dump(target_enc, 'target_encoder.joblib')

# Save the fitted StandardScaler
joblib.dump(scaler, 'scaler.joblib')

# Save the column order for the API
joblib.dump(X_train.columns.tolist(), 'model_columns.joblib')

print("Artifacts saved successfully. Script finished.")