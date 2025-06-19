import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The functions --- (load_config, build_lstm_autoencoder - remain the same)

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

def load_processed_data_parquet(file_path: Path) -> pd.DataFrame:
    """Uploads pre-processed data from the Parquet file."""
    if not file_path.exists():
        logging.error(f"Error: Parquet data file is not found on the path {file_path}")
        exit(1)
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Data from Parquet has been successfully downloaded: {file_path} Size: {df.shape}")
        # df.info(verbose=True, show_counts=True) # For more detailed information, you can comment
        return df
    except Exception as e:
        logging.error(f"Error when downloading Parquet data from {file_path}: {e}")
        exit(1)

def load_sequences_npy(file_path: Path) -> np.ndarray:
    """Loads sequences from the NPY file."""
    if not file_path.exists():
        logging.error(f"Error: NPY file is not found on the path {file_path}")
        exit(1)
    try:
        sequences = np.load(file_path)
        logging.info(f"The sequences from the NPY have been successfully downloaded: {file_path}. Shape: {sequences.shape}")
        return sequences
    except Exception as e:
        logging.error(f"Error when loading NPY file from {file_path}: {e}")
        exit(1)


def create_sequences_from_df(data_values: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Creates sequences (windows) from an array of time series values.
    For an autoencoder, X and y are the same sequence.
    """
    xs = []
    if len(data_values) < sequence_length:
        logging.warning(f"The data ({len(data_values)}) is smaller than the sequence length ({sequence_length}). You can't create sequences.")
        return np.array(xs) # Return the empty array if there is not enough data

    for i in range(len(data_values) - sequence_length + 1):
        x = data_values[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)

def build_lstm_autoencoder(sequence_length: int, num_features: int, config_params: dict) -> Model:
    """Builds an LSTM autoencoder model."""
    lstm_units_e1 = config_params.get('lstm_units_encoder1', 64)
    lstm_units_e2_latent = config_params.get('lstm_units_encoder2_latent', 32)
    lstm_units_d1 = config_params.get('lstm_units_decoder1', 32)
    lstm_units_d2 = config_params.get('lstm_units_decoder2', 64)

    inputs = Input(shape=(sequence_length, num_features))
    encoder = LSTM(lstm_units_e1, activation='relu', return_sequences=True)(inputs)
    encoder = LSTM(lstm_units_e2_latent, activation='relu', return_sequences=False)(encoder)
    bridge = RepeatVector(sequence_length)(encoder)
    decoder = LSTM(lstm_units_d1, activation='relu', return_sequences=True)(bridge)
    decoder = LSTM(lstm_units_d2, activation='relu', return_sequences=True)(decoder)
    outputs = TimeDistributed(Dense(num_features, activation='sigmoid'))(decoder)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_model(model_path: Path) -> tf.keras.Model:
    """Downloads the TensorFlow/Keras model from the file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"The model was successfully downloaded from {model_path}")
        model.summary(print_fn=logging.info)
        return model
    except Exception as e:
        logging.error(f"Error when loading the model from {model_path}: {e}", exc_info=True)
        raise

# --- Basic block ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    
    CONFIG = load_config(CONFIG_FILE_PATH)

    # Getting a pathway for artifacts ---
    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    # Create a directory if it does not exist.
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory for artifacts: {artifacts_dir}")

    # Removing settings
    preprocess_settings = CONFIG.get('preprocessing_settings', {})
    training_settings = CONFIG.get('training_settings', {})

    # General training parameters
    sequence_length = training_settings.get('sequence_length', 20)
    train_split_ratio = training_settings.get('train_split_ratio', 0.8)
    epochs = training_settings.get('epochs', 50)
    batch_size = training_settings.get('batch_size', 64)
    learning_rate = training_settings.get('learning_rate', 0.001)
    early_stopping_patience = training_settings.get('early_stopping_patience', 0)

    # Definition of input data and output model name
    logging.info("Training mode: on pre-processed data")
    input_parquet_filename = training_settings.get('input_processed_filename',
                                                  preprocess_settings.get('processed_output_filename'))
    if not input_parquet_filename:
        logging.error("The name of the file with the preprocessed Parquet data is not specified in config.yaml"
                      "(training_settings.input_processed_filename or preprocessing_settings.processed_output_filename.")
        exit(1)
        
    # Form the path to the input file inside the artifact directory.
    input_file_path = artifacts_dir / input_parquet_filename
    df_processed = load_processed_data_parquet(input_file_path)

    if df_processed.empty:
        logging.error("The downloaded DataFrame for training is empty. Training is impossible.")
        exit(1)

    num_features = df_processed.shape[1]
    data_values = df_processed.values
    X_all_sequences = create_sequences_from_df(data_values, sequence_length)
    model_output_filename = training_settings.get('model_output_filename', 'lstm_autoencoder_model.keras')

    if X_all_sequences.shape[0] == 0:
        logging.error("It was not possible to create any sequences from Parquet data. Training is impossible.")
        exit(1)

    # Form the path to the model output file inside the artifact directory.
    model_output_path = artifacts_dir / model_output_filename
    logging.info(f"The model will be saved to: {model_output_path}")

    # For an autoencoder, X (input) and Y (target) are the same sequences.
    y_all_sequences = X_all_sequences 
    logging.info(f"A total of {X_all_sequences.shape[0]} sequences with {X_all_sequences.shape[1]} features are created/downloaded.")

    # Separation of data into training and validation samples
    if X_all_sequences.shape[0] < 2 : # You need at least 2 samples to separate.
        logging.error(f"There are too few sequences ({X_all_sequences.shape[0]}) to divide into train/validation.")
        exit(1)

    val_split_size = 1.0 - train_split_ratio
    if val_split_size <= 0.0 or val_split_size >= 1.0:
        if X_all_sequences.shape[0] > 1: # If there's anything to validate
            logging.warning(f"train_split_ratio ({train_split_ratio}) It is incorrect to create a validation sample. A single sequence is used for validation if possible.")
            # If train_split_ratio = 1, then val_split_size = 0. sklearn requires test_size > 0
            # In this case, you can take 1 sample for validation, if all of them > 1.
            if X_all_sequences.shape[0] > 1 and train_split_ratio >= 1.0 :
                 X_train, X_val, y_train, y_val = X_all_sequences[:-1], X_all_sequences[-1:], y_all_sequences[:-1], y_all_sequences[-1:]
            elif X_all_sequences.shape[0] > 1 and train_split_ratio <= 0.0 : # All validated.
                 X_train, X_val, y_train, y_val = X_all_sequences[:1], X_all_sequences[1:], y_all_sequences[:1], y_all_sequences[1:]
            else: # If there's only one sample, he'll go to Train, no validation.
                 X_train, X_val, y_train, y_val = X_all_sequences, np.array([]).reshape(0,sequence_length,num_features), y_all_sequences, np.array([]).reshape(0,sequence_length,num_features)
                 logging.warning("The validation sample is empty because there is not enough data.")
        else:
            X_train, X_val, y_train, y_val = X_all_sequences, np.array([]).reshape(0,sequence_length,num_features), y_all_sequences, np.array([]).reshape(0,sequence_length,num_features)
            logging.warning("The validation sample is empty because there is only one sequence.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all_sequences, y_all_sequences, train_size=train_split_ratio, shuffle=True, random_state=42
        )

    logging.info(f"Training sample size: {X_train.shape}")
    logging.info(f"Validation sample size: {X_val.shape}")

    # Building an LSTM autoencoder model
    autoencoder_model = build_lstm_autoencoder(sequence_length, num_features, training_settings)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder_model.compile(optimizer=optimizer, loss='mse')
    autoencoder_model.summary(print_fn=logging.info)

    # Model learning
    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping)
        logging.info(f"EarlyStopping Included with patience: {early_stopping_patience} eras")
    
    # Directory for checkpoints inside the artifact directory
    checkpoint_dir = artifacts_dir / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = checkpoint_dir / "best_model.keras"
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)
    callbacks.append(model_checkpoint_callback)
    logging.info(f"ModelCheckpoint included. The best model will be saved in: {checkpoint_filepath}")

    logging.info("\nBegin model training. . .")
    validation_data_to_pass = None
    if X_val.shape[0] > 0:
        validation_data_to_pass = (X_val, y_val)

    history = autoencoder_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_to_pass,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    logging.info("Training is complete.")

    if checkpoint_filepath.exists() and any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        logging.info(f"Download the best model from checkpoint: {checkpoint_filepath}")
        try:
            autoencoder_model = load_model(checkpoint_filepath)
        except Exception as e:
            logging.error(f"Error loading model from checkpoint {checkpoint_filepath}: {e}. The latest model is maintained.")

    try:
        autoencoder_model.save(model_output_path)
        logging.info(f"The trained model is saved in: {model_output_path}")
    except Exception as e:
        logging.error(f"Error in saving the model: {e}")

    # Visualization of learning history (loss)
    if validation_data_to_pass: 
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Learning Loss (Train Loss)')
        plt.plot(history.history['val_loss'], label='Validation loss (validation loss)')
        plt.title(f'Learning history ({model_output_filename})')
        plt.xlabel('Epoch')
        plt.ylabel('Losses (MSE)')
        plt.legend()
        plt.grid(True)
        # The way to save the graph inside the artifact directory
        plot_filename_loss = artifacts_dir / f"training_history_loss_{model_output_filename.replace('.keras', '')}.png"
        try:
            plt.savefig(plot_filename_loss)
            logging.info(f"Graph of the history of learning is saved in: {plot_filename_loss}")
        except Exception as e:
            logging.error(f"Error in maintaining the schedule of learning history: {e}")
    else:
        logging.info("The val_loss graph is not constructed because the validation sample was empty.")

    # Evaluation of the distribution of reconstruction errors on validation data (if any)
    if X_val.shape[0] > 0:
        logging.info("\nEvaluation of reconstruction errors on the validation sample. . .")
        X_val_pred = autoencoder_model.predict(X_val, batch_size=batch_size)
        mse_val = np.mean(np.power(X_val - X_val_pred, 2), axis=(1, 2))

        plt.figure(figsize=(10, 6))
        plt.hist(mse_val, bins=50, density=True, alpha=0.75)
        plt.title(f'Histogram of reconstruction errors ({model_output_filename}) on validation (MSE)')
        plt.xlabel('Reconstruction error (MSE)')
        plt.ylabel('Density')
        plt.grid(True)
        # The way to save the histogram inside the artifact directory
        plot_filename_hist = artifacts_dir / f"reconstruction_error_histogram_{model_output_filename.replace('.keras', '')}.png"
        try:
            plt.savefig(plot_filename_hist)
            logging.info(f"The reconstruction error histogram is saved in: {plot_filename_hist}")
        except Exception as e:
            logging.error(f"The error in saving the error histogram: {e}")
    else:
        logging.info("A histogram of reconstruction errors on validation is not built, since the validation sample was empty.")

    logging.info(f"--- Model learning script'{model_output_filename}' completed--")