import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cookiecutter_project.data import MyDataset


def test_my_dataset_initialization():
    """Test that MyDataset initializes correctly."""
    dataset = MyDataset()
    
    # Check that all attributes exist
    assert hasattr(dataset, 'x_train')
    assert hasattr(dataset, 'x_test')
    assert hasattr(dataset, 'y_train')
    assert hasattr(dataset, 'y_test')
    assert hasattr(dataset, 'scaler')


def test_my_dataset_shapes():
    """Test that train/test splits have correct shapes."""
    dataset = MyDataset()
    
    # Breast cancer dataset has 569 samples, 80/20 split = 455 train, 114 test
    assert len(dataset.x_train) == 455
    assert len(dataset.x_test) == 114
    assert len(dataset.y_train) == 455
    assert len(dataset.y_test) == 114
    
    # Features: breast cancer has 30 features
    assert dataset.x_train.shape[1] == 30
    assert dataset.x_test.shape[1] == 30


def test_my_dataset_data_types():
    """Test that data is numpy arrays."""
    dataset = MyDataset()
    
    assert isinstance(dataset.x_train, np.ndarray)
    assert isinstance(dataset.x_test, np.ndarray)
    assert isinstance(dataset.y_train, np.ndarray)
    assert isinstance(dataset.y_test, np.ndarray)


def test_my_dataset_labels_1d():
    """Test that labels are 1D arrays for sklearn compatibility."""
    dataset = MyDataset()
    
    assert dataset.y_train.ndim == 1
    assert dataset.y_test.ndim == 1


def test_my_dataset_scaled():
    """Test that features are scaled (mean ~0, std ~1)."""
    dataset = MyDataset()
    
    # Check that training data is scaled (mean close to 0, std close to 1)
    train_mean = np.mean(dataset.x_train, axis=0)
    train_std = np.std(dataset.x_train, axis=0)
    
    # Mean should be very close to 0 (within tolerance)
    assert np.allclose(train_mean, 0, atol=1e-10)
    # Std should be very close to 1 (within tolerance)
    assert np.allclose(train_std, 1, atol=1e-10)


def test_my_dataset_len():
    """Test the __len__ method."""
    dataset = MyDataset()
    
    assert len(dataset) == 455  # Training set size
    assert len(dataset) == len(dataset.x_train)


def test_my_dataset_getitem():
    """Test the __getitem__ method returns correct format."""
    dataset = MyDataset()
    
    # Get first item
    x, y = dataset[0]
    
    # Should return tensors
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    
    # Check shapes
    assert x.shape == (30,)  # 30 features
    assert y.shape == (1,)  # Single label
    
    # Check data types
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_my_dataset_getitem_multiple():
    """Test __getitem__ with different indices."""
    dataset = MyDataset()
    
    # Test first item
    x1, y1 = dataset[0]
    assert x1.shape == (30,)
    assert y1.shape == (1,)
    
    # Test middle item
    x2, y2 = dataset[100]
    assert x2.shape == (30,)
    assert y2.shape == (1,)
    
    # Test last item
    x3, y3 = dataset[len(dataset) - 1]
    assert x3.shape == (30,)
    assert y3.shape == (1,)


def test_my_dataset_getitem_index_error():
    """Test that __getitem__ raises IndexError for out of bounds."""
    dataset = MyDataset()
    
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]  # Out of bounds
    
    # Negative indexing should work (Python supports it)
    # Test that it returns the last element
    x_neg, y_neg = dataset[-1]
    x_last, y_last = dataset[len(dataset) - 1]
    assert torch.equal(x_neg, x_last)
    assert torch.equal(y_neg, y_last)


def test_my_dataset_preprocess():
    """Test the preprocess method saves files correctly."""
    dataset = MyDataset()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir) / "processed"
        
        # Run preprocessing
        dataset.preprocess(output_folder)
        
        # Check that files were created
        assert (output_folder / "x_train.pt").exists()
        assert (output_folder / "x_test.pt").exists()
        assert (output_folder / "y_train.pt").exists()
        assert (output_folder / "y_test.pt").exists()
        
        # Check that files can be loaded
        x_train_loaded = torch.load(output_folder / "x_train.pt")
        x_test_loaded = torch.load(output_folder / "x_test.pt")
        y_train_loaded = torch.load(output_folder / "y_train.pt")
        y_test_loaded = torch.load(output_folder / "y_test.pt")
        
        # Check shapes match
        assert x_train_loaded.shape == dataset.x_train.shape
        assert x_test_loaded.shape == dataset.x_test.shape
        assert y_train_loaded.shape == dataset.y_train.shape
        assert y_test_loaded.shape == dataset.y_test.shape


def test_my_dataset_reproducibility():
    """Test that dataset is reproducible (same random_state)."""
    dataset1 = MyDataset()
    dataset2 = MyDataset()
    
    # With same random_state, splits should be identical
    np.testing.assert_array_equal(dataset1.x_train, dataset2.x_train)
    np.testing.assert_array_equal(dataset1.x_test, dataset2.x_test)
    np.testing.assert_array_equal(dataset1.y_train, dataset2.y_train)
    np.testing.assert_array_equal(dataset1.y_test, dataset2.y_test)


def test_my_dataset_label_values():
    """Test that labels are binary (0 or 1) for breast cancer dataset."""
    dataset = MyDataset()
    
    # Breast cancer is binary classification
    unique_train = np.unique(dataset.y_train)
    unique_test = np.unique(dataset.y_test)
    
    assert set(unique_train).issubset({0, 1})
    assert set(unique_test).issubset({0, 1})


def test_my_dataset_no_nan_values():
    """Test that dataset contains no NaN or infinite values."""
    dataset = MyDataset()
    
    assert not np.isnan(dataset.x_train).any()
    assert not np.isnan(dataset.x_test).any()
    assert not np.isnan(dataset.y_train).any()
    assert not np.isnan(dataset.y_test).any()
    
    assert not np.isinf(dataset.x_train).any()
    assert not np.isinf(dataset.x_test).any()
    assert not np.isinf(dataset.y_train).any()
    assert not np.isinf(dataset.y_test).any()


def test_my_dataset_test_scaled_with_train_scaler():
    """Test that test data is scaled using the training scaler."""
    dataset = MyDataset()
    
    # Test data should be transformed with the scaler fitted on training data
    # This means test data mean/std won't be exactly 0/1, but should be reasonable
    test_mean = np.mean(dataset.x_test, axis=0)
    test_std = np.std(dataset.x_test, axis=0)
    
    # Test mean should be close to 0 (but not exactly due to different distribution)
    assert np.allclose(test_mean, 0, atol=1.0)
    # Test std should be close to 1
    assert np.allclose(test_std, 1, atol=1.0)


def test_preprocess_function():
    """Test that the preprocess function saves data files correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir) / "processed"
        
        # Import and call the preprocess function
        from cookiecutter_project.data import preprocess
        
        # Call the function
        preprocess(str(output_folder))
        
        # Check that all files were created
        assert (output_folder / "x_train.pt").exists(), \
            f"preprocess should create x_train.pt, but file does not exist at {output_folder / 'x_train.pt'}"
        assert (output_folder / "x_test.pt").exists(), \
            f"preprocess should create x_test.pt, but file does not exist at {output_folder / 'x_test.pt'}"
        assert (output_folder / "y_train.pt").exists(), \
            f"preprocess should create y_train.pt, but file does not exist at {output_folder / 'y_train.pt'}"
        assert (output_folder / "y_test.pt").exists(), \
            f"preprocess should create y_test.pt, but file does not exist at {output_folder / 'y_test.pt'}"
        
        # Verify files can be loaded and have correct shapes
        x_train_loaded = torch.load(output_folder / "x_train.pt")
        x_test_loaded = torch.load(output_folder / "x_test.pt")
        y_train_loaded = torch.load(output_folder / "y_train.pt")
        y_test_loaded = torch.load(output_folder / "y_test.pt")
        
        assert x_train_loaded.shape[0] == 455, \
            f"x_train should have 455 samples, but got {x_train_loaded.shape[0]}"
        assert x_test_loaded.shape[0] == 114, \
            f"x_test should have 114 samples, but got {x_test_loaded.shape[0]}"
        assert y_train_loaded.shape[0] == 455, \
            f"y_train should have 455 samples, but got {y_train_loaded.shape[0]}"
        assert y_test_loaded.shape[0] == 114, \
            f"y_test should have 114 samples, but got {y_test_loaded.shape[0]}"
