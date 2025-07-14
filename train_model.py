import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics

# 1. Load data with path verification
data_path = os.path.join("data", "census.csv")
data = pd.read_csv(data_path)

# 2. Clean column names (remove spaces)
data.columns = [col.strip() for col in data.columns]

# 3. Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# 4. Process data with proper train-test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# 5. Train model with verification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model
preds = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("\n=== Model Performance ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {fbeta:.4f}")

# 7. Save artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

print("\n=== Files Saved ===")
print("model.pkl, encoder.pkl, lb.pkl saved to model/ directory")
