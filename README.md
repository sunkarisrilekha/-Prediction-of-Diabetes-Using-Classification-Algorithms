import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample Diabetes Dataset
data = {
    'Glucose': [120, 85, 150, 95, 140, 110, 130, 100],
    'BloodPressure': [70, 65, 80, 72, 85, 75, 78, 68],
    'BMI': [30, 25, 35, 28, 33, 27, 31, 26],
    'Age': [45, 30, 50, 29, 48, 35, 42, 31],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom Prediction
sample_data = [[135, 80, 32, 40]]

sample_scaled = scaler.transform(sample_data)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("\nPrediction: Diabetic")
else:
    print("\nPrediction: Not Diabetic")

