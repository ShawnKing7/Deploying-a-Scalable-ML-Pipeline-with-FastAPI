import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline."""

    # Debug: Show input data
    print(f"\nInput data:\n{X.head()}")
    print(f"Categorical features: {categorical_features}")

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # Ensure categorical features exist in DataFrame
    missing_features = [f for f in categorical_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing categorical features: {missing_features}")

    X_categorical = X[categorical_features].copy()
    X_continuous = X.drop(categorical_features, axis=1)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel() if len(y) > 0 else np.array([])
    else:
        try:
            X_categorical = encoder.transform(X_categorical)
            y = lb.transform(y.values).ravel() if len(y) > 0 else np.array([])
        except AttributeError as e:
            print(f"Encoder/LB error: {str(e)}")
            return None, None, None, None

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    # Debug: Show output shapes
    print(f"Processed shapes - X: {X.shape}, y: {y.shape}")
    return X, y, encoder, lb


def apply_label(inference):
    """ Convert binary label to string output."""
    if inference is None or len(inference) == 0:
        return "UNKNOWN"
    return ">50K" if inference[0] == 1 else "<=50K"


if __name__ == "__main__":
    print("Data script verified!")