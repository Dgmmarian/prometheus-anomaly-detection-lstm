import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib # To save and load the scaler
import numpy as np # For np.nan if necessary

# --- Functions ---

def load_config(config_path: Path) -> dict:
    """Downloads the configuration from the YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        print(f"Configuration successfully downloaded from {config_path}")
        return config_data
    except FileNotFoundError:
        print(f"Error: The configuration file is not found on the path {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)
    except Exception as e:
        print(f"Unforeseen error when loading configuration {config_path}: {e}")
        exit(1)

def load_data(file_path: Path) -> pd.DataFrame:
    """Uploads data from the Parquet file."""
    if not file_path.exists():
        print(f"Error: The data file is not found on the path {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        print(f"The data is successfully downloaded from {file_path}. Size: {df.shape}")
        df.info()
        return df
    except Exception as e:
        print(f"Error when loading data from {file_path}: {e}")
        exit(1)

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Processes missing values in DataFrame."""
    print(f"\nProcessing missed values (NaN) by strategy: {strategy}")
    print(f"The amount of NaN before processing:\n{df.isnull() .sum()")

    if strategy == "ffill_then_bfill":
        df_filled = df.ffill().bfill()
    elif strategy == "mean":
        # Make sure we apply only to numerical columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                print(f"Warning: Column {col} is not numerical, pass for'mean' fill.")
        df_filled = df # df changed in place
    elif strategy == "median":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                 print(f"Warning: Column {col} is not numerical, pass for'median' fill.")
        df_filled = df # df changed in place
    elif strategy == "drop_rows":
        df_filled = df.dropna()
    elif strategy == "none":
        print("NaN omissions were not handled according to configuration.")
        df_filled = df
    else:
        print(f"Warning: Unknown NaN filling strategy'{strategy}'. Passes untreated.")
        df_filled = df
    
    print(f"The amount of NaN after processing:\n{df_filled.isnull().sum()}")
    if df_filled.isnull().sum().sum() > 0:
        print("Attention: After processing, there are still NaN values left. Check the data and the strategy.")
    return df_filled

def scale_data(df: pd.DataFrame, scaler_type: str, scaler_output_path: Path) -> (pd.DataFrame, object):
    """Scale data and save the scaler."""
    print(f"\nScaling data using: {scaler_type}")
    # All columns in df are metrics that need to be scaled.
    # If you have non-metric columns (other than an index), you need to filter them out.
    metric_columns = df.columns
    data_to_scale = df[metric_columns].values # Get a NumPy array for the scaler

    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        print(f"Warning: Unknown type of skater'{scaler_type}'. MinMaxScaler is used by default.")
        scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data_to_scale)
    df_scaled = pd.DataFrame(scaled_data, columns=metric_columns, index=df.index)

    try:
        joblib.dump(scaler, scaler_output_path)
        print(f"The scaler is saved in: {scaler_output_path}")
    except Exception as e:
        print(f"Error in saving the scaler: {e}")

    return df_scaled, scaler

def save_processed_data(df: pd.DataFrame, file_path: Path):
    """Stores the processed DataFrame in a Parquet file."""
    try:
        df.to_parquet(file_path, index=True)
        print(f"\nThe processed data is successfully stored in: {file_path}")
        df.info()
    except Exception as e:
        print(f"Error in saving processed data in {file_path}: {e}")

# --- Basic block ---
if __name__ == "__main__":
    # Determine the path to the configuration file
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Getting a pathway for artifacts ---
    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    # Create a directory if it doesn’t exist (just in case the data_collector should have already)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory for artifacts: {artifacts_dir}")

    # Receive settings from the configuration
    data_settings = CONFIG.get('data_settings', {})
    preprocess_settings = CONFIG.get('preprocessing_settings', {})

    # Input file (the result of the previous script)
    input_filename = preprocess_settings.get('input_filename', data_settings.get('output_filename'))
    if not input_filename:
        print("Mistake: The name of the input file is not specified in'preprocessing_settings.input_filename', neither'data_settings.output_filename'.")
        exit(1)
    
    # Form the path to the input file inside the artifact directory.
    input_file_path = artifacts_dir / input_filename

    # Pre-processing settings
    nan_strategy = preprocess_settings.get('nan_fill_strategy', 'ffill_then_bfill')
    scaler_type_config = preprocess_settings.get('scaler_type', 'MinMaxScaler')
    
    # Output files
    processed_output_filename = preprocess_settings.get('processed_output_filename', 'processed_metrics_data.parquet')
    scaler_output_filename = preprocess_settings.get('scaler_output_filename', 'fitted_scaler.joblib')

    # Form paths to output files inside the artifact directory
    processed_output_file_path = artifacts_dir / processed_output_filename
    scaler_output_file_path = artifacts_dir / scaler_output_filename

    print("--- Beginning of the data preprocessing script --")

    # 1. Downloading data
    raw_df = load_data(input_file_path)
    if raw_df.empty:
        print("The downloaded DataFrame is empty. Pretreatment is impossible.")
        exit(1)

    # 2. Processing of missing values
    # Copy DataFrame to avoid SettingWithCopyWarning when modified
    df_processed = raw_df.copy()
    df_processed = handle_missing_values(df_processed, nan_strategy)

    # Add signs of the day of the week and hour of the day
    df_processed['day_of_week'] = df_processed.index.dayofweek.astype(int)
    df_processed['hour_of_day'] = df_processed.index.hour.astype(int)
    
    if df_processed.empty and not raw_df.empty:
        print("DataFrame became empty after processing NaN (for example, due to drop_rows). Completion.")
        exit(1)
    if df_processed.isnull().values.any():
         print("WARNING: The data still has NaN after the pass processing phase. This can cause problems with scaling or learning.")


    # 3. Data scaling
    df_scaled, fitted_scaler = scale_data(df_processed, scaler_type_config, scaler_output_file_path)
    print("\nThe first 5 lines of scaled data:")
    print(df_scaled.head())

    # 4. Preservation of processed data
    save_processed_data(df_scaled, processed_output_file_path)
    
    print("\n--- The pre-processing script is completed -")