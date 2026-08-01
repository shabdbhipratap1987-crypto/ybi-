# Diabetes Prediction using Logistic Regression

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
url = "https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv"
diabetes = pd.read_csv(url)

# Step 3: Define Features and Target
X = diabetes.drop("diabetes", axis=1)
y = diabetes["diabetes"]

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.7,
    random_state=2529
)

# Step 5: Create Model
model = LogisticRegression(max_iter=500)

# Step 6: Train Model
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate Model
print("=" * 50)
print("Accuracy Score")
print("=" * 50)
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print("=" * 50)
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print("=" * 50)
print(classification_report(y_test, y_pred))

# Step 9: Predict New Patient

new_patient = [[2, 120, 70, 20, 79, 30.5, 0.45, 29]]

prediction = model.predict(new_patient)

print("\nPrediction Result")
print("=" * 50)

if prediction[0] == 1:
    print("Patient is likely to have Diabetes.")
else:
    print("Patient is unlikely to have Diabetes.")
