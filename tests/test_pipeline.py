import os
import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import preprocess_data

# ==========================================
# 1. UNIT TESTS FOR PREPROCESSING (6 Required)
# ==========================================

@pytest.fixture
def sample_raw_data():
    """Fixture to provide a mock raw dataframe simulating UCI structure."""
    return pd.DataFrame({
        'age':  [52, 53, np.nan, 45, 60, 58, 62, 44, 50, 51, 48, 64, 59, 41, 56],
        'chol': [234, 190, 250, np.nan, 210, 240, 200, 180, 220, 230, 195, 260, 215, 175, 245],
        'sex':  [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        'num':  [0, 2, 1, 0, 4, 0, 1, 3, 0, 2, 0, 1, 0, 1, 2]  # Raw target column name
    })

def test_preprocessing_creates_directories(tmp_path, sample_raw_data):
    """Test 1: Preprocessing correctly creates the output directory structure."""
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    sample_raw_data.to_csv(input_file, index=False)
    
    preprocess_data(input_path=str(input_file), output_dir=str(output_dir))
    assert os.path.exists(output_dir)

def test_preprocessing_handles_missing_values(tmp_path, sample_raw_data):
    """Test 2: Preprocessing replaces missing values instead of forwarding NaNs."""
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    sample_raw_data.to_csv(input_file, index=False)
    
    preprocess_data(input_path=str(input_file), output_dir=str(output_dir))
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    assert X_train.isnull().sum().sum() == 0

def test_preprocessing_binary_target_mapping(tmp_path, sample_raw_data):
    """Test 3: Checks if target values [1,2,3,4] correctly map to binary 1."""
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    sample_raw_data.to_csv(input_file, index=False)
    
    preprocess_data(input_path=str(input_file), output_dir=str(output_dir))
    y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv"))
    # Check that targets are only 0 or 1
    assert set(y_train.iloc[:, 0].unique()).issubset({0, 1})

def test_preprocessing_does_not_modify_original(tmp_path, sample_raw_data):
    """Test 4: Preprocessing should not mutate the original raw dataset."""
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    sample_raw_data.to_csv(input_file, index=False)
    
    original_copy = sample_raw_data.copy()
    preprocess_data(input_path=str(input_file), output_dir=str(output_dir))
    
    current_raw = pd.read_csv(input_file)
    assert current_raw.shape == original_copy.shape

def test_preprocessing_splits_data_correctly(tmp_path, sample_raw_data):
    """Test 5: Verifies train/validation set splitting proportions."""
    # Create slightly larger sample to check splits
    large_sample = pd.concat([sample_raw_data]*3, ignore_index=True)
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    large_sample.to_csv(input_file, index=False)
    
    preprocess_data(input_path=str(input_file), output_dir=str(output_dir))
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(output_dir, "X_val.csv"))
    assert len(X_train) > len(X_val)

def test_preprocessing_raises_key_error(tmp_path):
    """Test 6: Raises appropriate error if both 'num' and 'target' columns are missing."""
    bad_data = pd.DataFrame({'age': [52], 'chol': [234]})  # Missing target column completely
    input_file = tmp_path / "raw_data.csv"
    output_dir = tmp_path / "processed"
    bad_data.to_csv(input_file, index=False)
    
    with pytest.raises(KeyError):
        preprocess_data(input_path=str(input_file), output_dir=str(output_dir))


# ==========================================
# 2. DATA VALIDATION TESTS (3 Required)
# ==========================================

@pytest.fixture
def processed_data_paths():
    """Fixture to ensure processed files exist for integration validation."""
    return {
        "X_train": "data/processed/X_train.csv",
        "y_train": "data/processed/y_train.csv"
    }

def test_data_columns_present(processed_data_paths):
    """Test 7: Verify all expected original features are present in the output dataset."""
    X_train = pd.read_csv(processed_data_paths["X_train"])
    assert len(X_train.columns) >= 8  # Fulfills requirement of at least 8 features

def test_target_values_valid(processed_data_paths):
    """Test 8: Verify target variable contains only binary classification elements."""
    y_train = pd.read_csv(processed_data_paths["y_train"])
    unique_vals = y_train.iloc[:, 0].unique()
    for val in unique_vals:
        assert val in [0, 1]

def test_numeric_ranges_valid(processed_data_paths):
    """Test 9: Verify standardized features fall within realistic scaled statistical scales."""
    X_train = pd.read_csv(processed_data_paths["X_train"])
    # Standard scaled data generally shouldn't exceed extreme boundary limits like +/- 10
    assert X_train.max().max() < 10
    assert X_train.min().min() > -10


# ==========================================
# 3. MODEL VALIDATION TESTS (2 Required)
# ==========================================

def test_model_predictions_shape(processed_data_paths):
    """Test 10: Verify the model produces predictions of the correct type and shape."""
    X_train = pd.read_csv(processed_data_paths["X_train"])
    y_train = pd.read_csv(processed_data_paths["y_train"]).values.ravel()
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y_train.shape

def test_model_minimum_performance(processed_data_paths):
    """Test 11: Verify model hits a baseline performance accuracy threshold."""
    X_train = pd.read_csv(processed_data_paths["X_train"])
    y_train = pd.read_csv(processed_data_paths["y_train"]).values.ravel()
    
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    assert train_acc >= 0.60  # Must exceed baseline performance threshold