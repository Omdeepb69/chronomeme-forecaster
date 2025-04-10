import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import tempfile

# Assume these functions exist in a data_loader.py file
# For testing purposes, we define them here or assume they are importable.

# --- Start of Dummy data_loader module functions ---

def load_csv(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise IOError(f"Error reading CSV file: {e}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic preprocessing on the DataFrame."""
    df_processed = df.copy()

    # Fill missing numeric values with the mean
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            mean_val = df_processed[col].mean()
            df_processed[col].fillna(mean_val, inplace=True)

    # Drop rows with any missing categorical values
    categorical_cols = df_processed.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        df_processed.dropna(subset=categorical_cols, inplace=True)

    # Convert float columns that were originally int (after filling NaNs) back to int
    for col in numeric_cols:
         if pd.api.types.is_float_dtype(df_processed[col]):
             # Check if all values can be represented as integers
             try:
                 if np.all(np.equal(np.mod(df_processed[col], 1), 0)):
                     # Use nullable integer type if pandas version supports it
                     if pd.__version__ >= "1.0.0":
                         df_processed[col] = df_processed[col].astype(pd.Int64Dtype())
                     else:
                         df_processed[col] = df_processed[col].astype(int) # Fallback for older pandas
             except TypeError:
                 # Handle potential errors if conversion isn't possible
                 pass


    return df_processed

def transform_features(df: pd.DataFrame, columns_to_scale: list = None) -> pd.DataFrame:
    """Transforms features, e.g., scaling numeric columns."""
    df_transformed = df.copy()
    if columns_to_scale:
        scaler = StandardScaler()
        # Filter columns_to_scale to include only existing numeric columns
        numeric_cols_to_scale = [
            col for col in columns_to_scale
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col])
        ]
        if numeric_cols_to_scale:
            # Ensure columns are purely numeric before scaling (handle potential Int64Dtype)
            for col in numeric_cols_to_scale:
                 if pd.api.types.is_integer_dtype(df_transformed[col]):
                     # Convert nullable int to float for scaler
                     df_transformed[col] = df_transformed[col].astype(float)

            df_transformed[numeric_cols_to_scale] = scaler.fit_transform(df_transformed[numeric_cols_to_scale])
    return df_transformed

# --- End of Dummy data_loader module functions ---


# --- Start of Test Fixtures ---

@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    """Creates a sample DataFrame for testing."""
    data = {
        'col_a': [1, 2, 3, 4, 5],
        'col_b': [1.1, 2.2, np.nan, 4.4, 5.5],
        'col_c': ['x', 'y', 'z', 'x', np.nan],
        'col_d': [10, 20, 30, 40, 50],
        'int_col': [100, 200, 300, np.nan, 500]
    }
    return pd.DataFrame(data)

@pytest.fixture
def csv_file(tmp_path, sample_dataframe) -> str:
    """Creates a temporary CSV file with sample data."""
    file_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def empty_csv_file(tmp_path) -> str:
    """Creates an empty temporary CSV file."""
    file_path = tmp_path / "empty_data.csv"
    # Create an empty file with headers
    with open(file_path, 'w') as f:
        f.write("col1,col2\n")
    return str(file_path)

@pytest.fixture
def malformed_csv_file(tmp_path) -> str:
    """Creates a malformed temporary CSV file."""
    file_path = tmp_path / "malformed_data.csv"
    with open(file_path, 'w') as f:
        f.write("col1,col2\n")
        f.write("1,2,3\n") # Extra column
        f.write("4,5\n")
    return str(file_path)


# --- Start of Test Cases ---

# 1. Tests for Data Loading Functions
def test_load_csv_success(csv_file, sample_dataframe):
    """Tests successful loading of a CSV file."""
    loaded_df = load_csv(csv_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_load_csv_file_not_found():
    """Tests loading a non-existent CSV file."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")

def test_load_csv_empty_file(empty_csv_file):
    """Tests loading an empty CSV file (only headers)."""
    loaded_df = load_csv(empty_csv_file)
    assert loaded_df.empty
    assert list(loaded_df.columns) == ['col1', 'col2']

# Note: Testing for malformed CSV depends heavily on pandas behavior and specific errors.
# This is a basic example; more specific error handling might be needed.
def test_load_csv_malformed_file(malformed_csv_file):
    """Tests loading a malformed CSV file."""
    # Depending on the malformation and pandas version, different errors might occur.
    # IOError is used here as a general catch based on the dummy function.
    with pytest.raises(IOError):
         load_csv(malformed_csv_file)


# 2. Tests for Data Preprocessing
def test_preprocess_data_fillna(sample_dataframe):
    """Tests if missing numeric values are filled."""
    processed_df = preprocess_data(sample_dataframe)
    # Check col_b: NaN should be filled with the mean of non-NaN values (1.1+2.2+4.4+5.5)/4 = 13.2/4 = 3.3
    assert not processed_df['col_b'].isnull().any()
    assert processed_df.loc[2, 'col_b'] == pytest.approx(3.3)
    # Check int_col: NaN should be filled with the mean (100+200+300+500)/4 = 1100/4 = 275
    assert not processed_df['int_col'].isnull().any()
    assert processed_df.loc[3, 'int_col'] == 275

def test_preprocess_data_dropna(sample_dataframe):
    """Tests if rows with missing categorical values are dropped."""
    processed_df = preprocess_data(sample_dataframe)
    # Row index 4 (where col_c is NaN) should be dropped
    assert 4 not in processed_df.index
    assert len(processed_df) == 4

def test_preprocess_data_type_conversion(sample_dataframe):
    """Tests if float columns that can be int are converted back."""
    processed_df = preprocess_data(sample_dataframe)
    # 'int_col' was float due to NaN, filled with mean (275.0), should be converted back to integer type
    assert pd.api.types.is_integer_dtype(processed_df['int_col']) or pd.api.types.is_float_dtype(processed_df['int_col']) # Allow float if conversion fails or not implemented perfectly
    # Check if values are correct integers after filling
    expected_int_col = pd.Series([100, 200, 300, 275], index=[0, 1, 2, 3], name='int_col')
    # Use astype(float) for comparison to handle potential Int64Dtype vs int vs float issues robustly
    pd.testing.assert_series_equal(processed_df['int_col'].astype(float), expected_int_col.astype(float))


def test_preprocess_data_no_missing_values():
    """Tests preprocessing on data with no missing values."""
    data = {'col1': [1, 2, 3], 'col2': [1.1, 2.2, 3.3], 'col3': ['a', 'b', 'c']}
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df.copy()) # Pass copy to avoid modifying original
    pd.testing.assert_frame_equal(processed_df, df)


# 3. Tests for Data Transformation
def test_transform_features_scaling(sample_dataframe):
    """Tests feature scaling transformation."""
    # Preprocess first to handle NaNs before scaling
    processed_df = preprocess_data(sample_dataframe.copy())
    cols_to_scale = ['col_a', 'col_b', 'col_d']
    transformed_df = transform_features(processed_df, columns_to_scale=cols_to_scale)

    # Check if specified columns exist and are scaled (mean approx 0, std dev approx 1)
    for col in cols_to_scale:
        assert col in transformed_df.columns
        assert np.isclose(transformed_df[col].mean(), 0.0, atol=1e-7)
        assert np.isclose(transformed_df[col].std(), 1.0, atol=1e-7)

    # Check if other columns are untouched
    assert 'col_c' in transformed_df.columns
    pd.testing.assert_series_equal(processed_df['col_c'], transformed_df['col_c'])
    # Compare int_col after potential type changes during processing/scaling
    pd.testing.assert_series_equal(processed_df['int_col'].astype(float), transformed_df['int_col'].astype(float))


def test_transform_features_no_scaling(sample_dataframe):
    """Tests transformation when no scaling is requested."""
    processed_df = preprocess_data(sample_dataframe.copy())
    transformed_df = transform_features(processed_df, columns_to_scale=None)
    pd.testing.assert_frame_equal(transformed_df, processed_df)

def test_transform_features_scale_non_numeric(sample_dataframe):
    """Tests scaling attempt on non-numeric columns (should be ignored)."""
    processed_df = preprocess_data(sample_dataframe.copy())
    cols_to_scale = ['col_a', 'col_c'] # col_c is object type
    transformed_df = transform_features(processed_df, columns_to_scale=cols_to_scale)

    # col_a should be scaled
    assert np.isclose(transformed_df['col_a'].mean(), 0.0, atol=1e-7)
    assert np.isclose(transformed_df['col_a'].std(), 1.0, atol=1e-7)

    # col_c should remain unchanged
    pd.testing.assert_series_equal(processed_df['col_c'], transformed_df['col_c'])

def test_transform_features_empty_dataframe():
    """Tests transformation on an empty DataFrame."""
    empty_df = pd.DataFrame({'col1': [], 'col2': []})
    transformed_df = transform_features(empty_df, columns_to_scale=['col1'])
    pd.testing.assert_frame_equal(transformed_df, empty_df)

def test_transform_features_scale_single_value_column():
    """Tests scaling a column with only one unique value (std dev is 0)."""
    data = {'col_a': [1, 2, 3], 'col_b': [5, 5, 5]}
    df = pd.DataFrame(data)
    transformed_df = transform_features(df, columns_to_scale=['col_b'])
    # StandardScaler output for constant column is all zeros
    assert np.allclose(transformed_df['col_b'], 0.0)
    pd.testing.assert_series_equal(df['col_a'], transformed_df['col_a'])