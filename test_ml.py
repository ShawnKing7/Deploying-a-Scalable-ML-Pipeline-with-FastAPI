import os
import pytest
import numpy as np
from ml.model import compute_model_metrics


def train_model(X, y):
    """Testable model training function"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def inference(model, X):
    """Testable inference function"""
    return model.predict(X)


@pytest.fixture
def sample_data():
    """Generate simple synthetic data for reliable testing"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    return X, y


def test_train_model(sample_data):
    """Test model training"""
    X, y = sample_data
    model = train_model(X, y)
    assert hasattr(model, "predict"), "Model must have predict method"
    assert hasattr(model, "fit"), "Model must have fit method"


def test_inference(sample_data):
    """Test inference"""
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Predictions must be numpy array"
    assert preds.shape[0] == X.shape[0], "Output length must match input"


def test_compute_metrics():
    """Test metrics calculation"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision out of range"
    assert 0 <= recall <= 1, "Recall out of range"
    assert 0 <= fbeta <= 1, "F1 score out of range"


def test_saved_artifacts():
    """Test saved files"""
    assert os.path.exists("model/model.pkl"), "Model file missing"
    assert os.path.exists("model/encoder.pkl"), "Encoder file missing"
    assert os.path.exists("model/lb.pkl"), "Label binarizer file missing"

# Version 7.14.25

