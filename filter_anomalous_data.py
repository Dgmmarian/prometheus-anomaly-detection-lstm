import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions ---

def load_config(config_path: Path) -> dict:
    """Downloads the configuration from the YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logging.info(f"Configuration successfully downloaded from {config_path}")
        return config_data
    except FileNotFoundError:
        logging.error(f"Error: The configuration file is not found on the path {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        exit(1)
    return {}

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Uploads pre-processed data from the Parquet file."""
    if not file_path.exists():
        logging.error(f"Error: The data file is not found on the path {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"The pre-processed data is successfully downloaded from {file_path}. Size: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error when loading data from {file_path}: {e}")
        exit(1)

def load_tf_model(model_path: Path):
    """Loads a trained TensorFlow/Keras model."""
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        exit(1)
    try:
        model = load_model(model_path) # compile=False can be added if no further training is planned
        logging.info(f"The model was successfully downloaded from {model_path}")
        model.summary(print_fn=logging.info)
        return model
    except Exception as e:
        logging.error(f"The model loading error from {model_path}: {e}", exc_info=True)
        exit(1)

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """Creates sequences (windows) from the time series."""
    xs = []
    if len(data) < sequence_length:
        logging.warning(f"The data ({len(data)}) is smaller than the sequence length ({sequence_length}). You can't create sequences.")
        return np.array(xs)
        
    for i in range(len(data) - sequence_length + 1):
        x = data[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)

# --- Basic block ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Getting a pathway for artifacts ---
    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    # Create a directory in case the script runs separately
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory for artifacts: {artifacts_dir}")

    # Removing the necessary settings from different sections
    preprocess_settings = CONFIG.get('preprocessing_settings', {})
    training_settings = CONFIG.get('training_settings', {})
    rt_detection_settings = CONFIG.get('real_time_anomaly_detection', {})
    filtering_settings = CONFIG.get('data_filtering_settings', {})

    input_processed_filename = preprocess_settings.get('processed_output_filename')
    model_filename = training_settings.get('model_output_filename')
    sequence_length = training_settings.get('sequence_length')
    anomaly_threshold_mse = rt_detection_settings.get('anomaly_threshold_mse')

    if not all([input_processed_filename, model_filename, sequence_length, anomaly_threshold_mse]):
        logging.error("Not all necessary parameters (processed_output_filename, model_output_filename)"
                      "sequence_length, anomaly_threshold_mse) found in config.yaml. Check the sections."
                      "preprocessing_settings, training_settings, real_time_anomaly_detection.")
        exit(1)
    
    # Form paths to input files inside the artifact directory
    input_file_path = artifacts_dir / input_processed_filename
    model_path = artifacts_dir / model_filename

    # Output file names
    normal_seq_output_filename = filtering_settings.get('normal_sequences_output_filename', 'filtered_normal_sequences.npy')
    anomalous_seq_output_filename = filtering_settings.get('anomalous_sequences_output_filename', 'filtered_anomalous_sequences.npy')
    
    # Form paths to output files inside the artifact directory
    normal_seq_output_path = artifacts_dir / normal_seq_output_filename
    anomalous_seq_output_path = artifacts_dir / anomalous_seq_output_filename
    
    logging.info("--- Beginning of anomalous data filtering script")

    # 1. Download pre-processed data (scaled)
    df_processed = load_processed_data(input_file_path)
    if df_processed.empty:
        logging.error("The downloaded DataFrame is empty. Filtration is impossible.")
        exit(1)
    
    data_values = df_processed.values # NumPy stratum

    # 2. Loading a trained model
    model = load_tf_model(model_path)

    # 3. Sequence creation
    all_sequences = create_sequences(data_values, sequence_length)
    if all_sequences.shape[0] == 0:
        logging.error("It was not possible to create any sequence from the data. Check the data length and sequence_length.")
        exit(1)
    logging.info(f"Created {all_sequences.shape[0]} sequences for analysis.")

    # 4. Obtaining reconstructions and calculating MSE errors for each sequence
    logging.info("Receiving reconstructions from the model. .")
    reconstructed_sequences = model.predict(all_sequences, batch_size=training_settings.get('batch_size', 64))
    
    # Calculating the MSE for each sequence
    # all_sequences and reconstructed_sequences are shaped (num_samples, sequence_length, num_features)
    ms_errors = np.mean(np.power(all_sequences - reconstructed_sequences, 2), axis=(1, 2))
    logging.info(f"Reconstruction errors are calculated for {len(ms_errors)} sequences.")

    # 5. Threshold-based sequence filtration
    normal_mask = ms_errors <= anomaly_threshold_mse
    anomalous_mask = ms_errors > anomaly_threshold_mse

    normal_sequences = all_sequences[normal_mask]
    anomalous_sequences = all_sequences[anomalous_mask]

    num_normal = normal_sequences.shape[0]
    num_anomalous = anomalous_sequences.shape[0]
    total_sequences = all_sequences.shape[0]

    logging.info(f"--- Filtration results ----")
    logging.info(f"MSE threshold used: {anomaly_threshold_mse:.6f}")
    logging.info(f"All sequences: {total_sequences}")
    logging.info(f"Normal sequences: {num_normal} ({num_normal/total_sequences:.2%})")
    logging.info(f"Abnormal sequences: {num_anomalous} ({num_anomalous/total_sequences:.2%})")

    # 6. Preservation of filtered sequences
    try:
        np.save(normal_seq_output_path, normal_sequences)
        logging.info(f"Normal sequences are stored in: {normal_seq_output_path} (format: {normal_sequences.shape})")
    except Exception as e:
        logging.error(f"Error in maintaining normal sequences: {e}")

    try:
        np.save(anomalous_seq_output_path, anomalous_sequences)
        logging.info(f"Abnormal sequences are stored in: {anomalous_seq_output_path} (format: {anomalous_sequences.shape})")
    except Exception as e:
        logging.error(f"Error in preserving anomalous sequences: {e}")

    logging.info("--- Script of filtering anomalous data completed --")