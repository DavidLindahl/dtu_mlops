import os

import pickle
import numpy as np
import pytest

from cookiecutter_project.data import MyDataset


def test_model_predict_dimensions():
    """Test that the model's predict method returns the correct dimensions."""
    # 1. Get the directory where THIS test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one level (from 'tests/' to 'my_first_cookiecutter_project/')
    project_root = os.path.dirname(current_dir)

    # 3. Join the path properly
    model_path = os.path.join(project_root, "models", "knn_model.pkl")

    # Skip test if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Use actual dataset shape (30 features)
    x_test = np.random.rand(10, 30)
    y_pred = model.predict(x_test)

    # Should return one prediction per sample
    assert y_pred.shape == (10,), f"Expected shape (10,), got {y_pred.shape}"


def test_model_predict_binary_values():
    """Test that the model predicts binary values (0 or 1)."""
    # 1. Get the directory where THIS test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one level (from 'tests/' to 'my_first_cookiecutter_project/')
    project_root = os.path.dirname(current_dir)

    # 3. Join the path properly
    model_path = os.path.join(project_root, "models", "knn_model.pkl")

    # Skip test if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Use actual dataset for realistic test
    dataset = MyDataset()
    x_test = dataset.x_test[:10]  # First 10 test samples
    y_pred = model.predict(x_test)

    # Check shape
    assert y_pred.shape == (10,)

    # Check that all predictions are binary (0 or 1)
    unique_values = np.unique(y_pred)
    assert set(unique_values).issubset({0, 1}), f"Predictions should be 0 or 1, got {unique_values}"


def test_model_predict_with_full_dataset():
    """Test that the model can predict on the full test set."""
    # 1. Get the directory where THIS test file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up one level (from 'tests/' to 'my_first_cookiecutter_project/')
    project_root = os.path.dirname(current_dir)

    # 3. Join the path properly
    model_path = os.path.join(project_root, "models", "knn_model.pkl")

    # Skip test if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    dataset = MyDataset()
    y_pred = model.predict(dataset.x_test)

    # Check shape matches test set size
    assert y_pred.shape == dataset.y_test.shape, (
        f"Predictions shape {y_pred.shape} should match test labels shape {dataset.y_test.shape}"
    )

    # Check all predictions are binary
    assert set(np.unique(y_pred)).issubset({0, 1}), "All predictions should be binary (0 or 1)"

    # Check that we get some of each class (not all 0s or all 1s)
    unique_values = np.unique(y_pred)
    assert len(unique_values) >= 1, "Should have at least one predicted class"
