import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("breast-cancer.csv")

# Identify features and target
X = df.drop(columns=["diagnosis"])  # replace "diagnosis" with your target column name if different
y = df["diagnosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "model.joblib")

print("✅ Model trained and saved as model.joblib")
