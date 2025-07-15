import pandas as pd
from ml.data import process_data

# 1. Check data loading
data = pd.read_csv("data/census.csv")
print("Data columns:", data.columns.tolist())
print("\nFirst 3 rows:\n", data.head(3))

# 2. Check process_data
train = data.sample(frac=0.8, random_state=42)
cat_features = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race",
    "sex", "native-country"
]

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)
print("\nX_train shape:", X_train.shape)
print("y_train samples:", y_train[:5])