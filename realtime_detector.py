import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import time
import requests
from datetime import datetime, timedelta
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # For scaler

from prometheus_client import start_http_server, Gauge, Counter, REGISTRY

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variables for Prometheus metrics (corrected names)
PROM_LATEST_RECONSTRUCTION_ERROR_MSE = None
PROM_IS_ANOMALY_DETECTED = None
PROM_TOTAL_ANOMALIES_COUNT = None
PROM_FEATURE_RECONSTRUCTION_ERROR_MSE = None
PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS = None
PROM_DATA_POINTS_IN_CURRENT_WINDOW = None


class RealtimeAnomalyDetector:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

        # Getting a pathway for artifacts ---
        base_dir = Path(__file__).resolve().parent
        artifacts_path_str = self.config.get('artifacts_dir', 'artifacts')
        artifacts_dir = base_dir / artifacts_path_str
        logging.info(f"The directory for artifacts is used: {artifacts_dir}")

        self.prom_url = self.config.get('prometheus_url')
        self.queries = self.config.get('queries', {})
        # Metrics from Prometheus plus additional temporary signs
        self.time_feature_names = ['day_of_week', 'hour_of_day']
        self.metric_columns_ordered = list(self.queries.keys()) + self.time_feature_names

        rt_config = self.config.get('real_time_anomaly_detection', {})
        self.query_interval = rt_config.get('query_interval_seconds', 60)
        self.exporter_port = rt_config.get('exporter_port', 8001)
        self.metrics_prefix = rt_config.get(
            'metrics_prefix', 'anomaly_detector_')

        self.anomaly_threshold = rt_config.get('anomaly_threshold_mse', 0.0025)

        preprocess_config = self.config.get('preprocessing_settings', {})
        training_config = self.config.get('training_settings', {})

        scaler_filename = preprocess_config.get(
            'scaler_output_filename', 'fitted_scaler.joblib')
        model_a_filename = training_config.get(
            'model_output_filename', 'lstm_autoencoder_model.keras')
        self.sequence_length = training_config.get('sequence_length', 20)

        data_s_config = self.config.get('data_settings', {})
        self.data_step_duration_str = rt_config.get(
            'data_step_duration', data_s_config.get('step', '30s'))

        # Form paths to files inside the artifact directory
        self.scaler_path = artifacts_dir / scaler_filename
        self.model_a_path = artifacts_dir / model_a_filename
        
        self.scaler = self._load_scaler()
        logging.info("Load the model. . .")
        self.model_a = self._load_tf_model(self.model_a_path, "Model")

        if self.scaler:
            self.num_features = self.scaler.n_features_in_
            if self.num_features != len(self.metric_columns_ordered):
                logging.error(
                    f"Divergence in number of signs! {len(self.metric_columns_ordered)}"
                    f"{self.num_features}.")
        else:
            logging.error("Scaler's not loaded.")
            self.num_features = len(self.metric_columns_ordered)

        self._setup_prometheus_metrics()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logging.info(
                f"Configuration successfully downloaded from {self.config_path}")
            return config_data
        except Exception as e:
            logging.error(
                f"Configuration error {self.config_path}: {e}", exc_info=True)
            exit(1)

    def _load_scaler(self):
        if not self.scaler_path.exists():
            logging.error(f"Scaler file not found: {self.scaler_path}")
            return None
        try:
            scaler = joblib.load(self.scaler_path)
            logging.info(f"Scaler successfully downloaded from {self.scaler_path}")
            return scaler
        except Exception as e:
            logging.error(
                f"Error to download the scaler from {self.scaler_path}: {e}", exc_info=True)
            return None

    def _load_tf_model(self, model_path: Path, model_name_log: str):
        if not model_path.exists():
            logging.warning(
                f"Model file'{model_name_log}' not found: {model_path}")
            return None
        try:
            model = load_model(model_path)
            logging.info(f"Information about {model_name_log}:")
            model.summary(print_fn=logging.info)
            logging.info(f"{model_name_log} successfully downloaded from {model_path}")
            return model
        except Exception as e:
            logging.error(
                f"Download error {model_name_log} from {model_path}: {e}", exc_info=True)
            return None

    def _td_seconds(self, td_str: str) -> int:
        if td_str.endswith('s'):
            return int(td_str[:-1])
        if td_str.endswith('m'):
            return int(td_str[:-1]) * 60
        if td_str.endswith('h'):
            return int(td_str[:-1]) * 3600
        try:
            return int(td_str)
        except ValueError:
            logging.warning(
                f"The length of the step is not recognized: {td_str}. Use 30c.")
            return 30

    def _fetch_data_window(self) -> pd.DataFrame | None:
        if not self.prom_url or not self.queries:
            logging.error("URL Prometheus or requests undefined.")
            return None
        step_seconds = self._td_seconds(self.data_step_duration_str)
        window_duration_seconds = self.sequence_length * step_seconds
        now_time = datetime.now()
        # Round the end of the window to the nearest step boundary
        aligned_end_ts = int(now_time.timestamp()) // step_seconds * step_seconds
        end_time = datetime.fromtimestamp(aligned_end_ts)
        start_time_query = end_time - timedelta(
            seconds=window_duration_seconds + step_seconds * 2)
        all_metric_dfs = []
        logging.info(
            f"Data request: {start_time_query.strftime()'%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}, step {self.data_step_duration_str}")
        for custom_name in self.queries:
            query_string = self.queries[custom_name]
            api_url = f"{self.prom_url}/api/v1/query_range"
            params = {'query': query_string, 'start': start_time_query.timestamp(
            ), 'end': end_time.timestamp(), 'step': self.data_step_duration_str}
            try:
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    if len(data['data']['result']) > 0:
                        values = data['data']['result'][0].get('values', [])
                        if values:
                            df_metric = pd.DataFrame(values, columns=['timestamp', custom_name]).set_index(pd.to_datetime(
                                pd.DataFrame(values, columns=['timestamp', custom_name])['timestamp'], unit='s'))[[custom_name]]
                            df_metric[custom_name] = pd.to_numeric(
                                df_metric[custom_name], errors='coerce')
                            all_metric_dfs.append(df_metric)
                        else:
                            logging.warning(
                                f"No meaning for'{custom_name}'.")
                            return None
                    else:
                        logging.warning(
                            f"Empty'result' for'{custom_name}'.")
                        return None
                else:
                    logging.warning(
                        f"Failure for'{custom_name}': {data.get('errorType', '')} {data.get('error', data.get('status'))}")
                    return None
            except Exception as e:
                logging.error(
                    f"Mistake for'{custom_name}': {e}", exc_info=True)
                return None
        if not all_metric_dfs or len(all_metric_dfs) != len(self.queries):
            logging.warning("Not all metrics are loaded.")
            return None
        try:
            final_df = pd.concat(all_metric_dfs, axis=1, join='inner')
            # Adding temporary signs
            final_df['day_of_week'] = final_df.index.dayofweek.astype(int)
            final_df['hour_of_day'] = final_df.index.hour.astype(int)
            if len(final_df) >= self.sequence_length:
                if final_df.empty:
                    logging.warning("The final DataFrame is empty.")
                    return None
                try:
                    final_df = final_df[self.metric_columns_ordered]
                except KeyError as e:
                    logging.error(
                        f"Column order error: {e}. Waiting: {self.metric_columns_ordered}. Received: {final_df.columns.tolist()}")
                    return None
                    
                return final_df.tail(self.sequence_length)
            else:
                logging.warning(f"There is not enough data ({len(final_df)}) for the message ({self.sequence_length}).")
                if PROM_DATA_POINTS_IN_CURRENT_WINDOW:
                    PROM_DATA_POINTS_IN_CURRENT_WINDOW.set(len(final_df))
                return None
        except Exception as e:
            logging.error(f"DataFrame Combination Error: {e}", exc_info=True)
            return None

    def _preprocess_and_create_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.scaler is None:
            logging.error("Scaler's not loaded.")
            return None
        if df.isnull().values.any():
            logging.warning("NaN In the window, apply ffill(.bfill()")
            df = df.sort_index().ffill().bfill()
            if df.isnull().values.any():
                logging.error(
                    f"NaN They stayed. Columns: {df.columns[df.isnull().any())].tolist()}")
                return None
        try:
            if df[self.metric_columns_ordered].shape[1] != self.num_features:
                logging.error(
                    f"Sign error. Expectation: {self.num_features}, is: {df[self.metric_columns_ordered].shape[1]}.")
                return None
            scaled_values = self.scaler.transform(
                df[self.metric_columns_ordered].values)
        except Exception as e:
            logging.error(
                f"Scaling error: {e}. Data (form {df[self.metric_columns_ordered].shape}):\n{df[self.metric_columns_ordered].head(2)}")
            return None
        return np.expand_dims(scaled_values, axis=0)

    def _setup_prometheus_metrics(self):
        global PROM_LATEST_RECONSTRUCTION_ERROR_MSE, PROM_IS_ANOMALY_DETECTED, \
            PROM_TOTAL_ANOMALIES_COUNT, PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS, \
            PROM_DATA_POINTS_IN_CURRENT_WINDOW, PROM_FEATURE_RECONSTRUCTION_ERROR_MSE

        model_id_label_list = []
        feature_error_label_list = ['feature_name']
        metric_definitions = {
            'latest_reconstruction_error_mse': ('MSE Reconstruction error for the last window', model_id_label_list, Gauge),
            'is_anomaly_detected': ('Flag of anomaly (1 if anomaly, 0 if normal)', model_id_label_list, Gauge),
            'total_anomalies_count': ('Total number of anomalies detected', model_id_label_list, Counter),
            'feature_reconstruction_error_mse': ('MSE error for an individual feature in the last window', feature_error_label_list, Gauge),
            'last_successful_run_timestamp_seconds': ('Timestamp The latest successful detection cycle', [], Gauge),
            'data_points_in_current_window': ('Number of data points in the current analyzed window', [], Gauge)
        }
        for name_suffix, (doc, labels, metric_type) in metric_definitions.items():
            full_name = self.metrics_prefix + name_suffix
            # Use corrected names of global variables
            global_var_name = f"PROM_{name_suffix.upper()}"
            if globals().get(global_var_name) is not None:  # Checking Existence and Meaning
                if full_name in REGISTRY._names_to_collectors:
                    try:
                        REGISTRY.unregister(
                            REGISTRY._names_to_collectors[full_name])
                        logging.info(f"Deleted old metric: {full_name}")
                    except Exception as e:
                        logging.warning(
                            f"The old metric {full_name} cannot be deleted: {e}")
            current_labels = tuple(labels) if labels else ()
            if metric_type == Gauge:
                globals()[global_var_name] = Gauge(
                    full_name, doc, labelnames=current_labels)
            elif metric_type == Counter:
                globals()[global_var_name] = Counter(
                    full_name, doc, labelnames=current_labels)
        logging.info("Prometheus The metrics are initialized.")
        if PROM_TOTAL_ANOMALIES_COUNT:
            try:
                PROM_TOTAL_ANOMALIES_COUNT.inc(0)
                logging.info("Total_anomalies_count is initialized.")
            except Exception as e:
                logging.warning(f"Failed to initialize the counter: {e}")

    def _process_model_output(self, model, sequence_to_predict: np.ndarray, threshold: float):
        # Use corrected names for global variable metrics
        if model is None:
            logging.warning("The model isn't loaded.")
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE:
                PROM_LATEST_RECONSTRUCTION_ERROR_MSE.set(0)
            if PROM_IS_ANOMALY_DETECTED:
                PROM_IS_ANOMALY_DETECTED.set(0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for fnk in self.metric_columns_ordered:
                    PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(
                        feature_name=fnk).set(0)
            return
        try:
            reconstructed_sequence = model.predict(
                sequence_to_predict, verbose=0)
            mse = np.mean(np.power(sequence_to_predict -
                          reconstructed_sequence, 2))
            logging.info(f"MSE: {mse:.6f}")
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE:
                PROM_LATEST_RECONSTRUCTION_ERROR_MSE.set(mse)

            sq_err_elems = np.power(
                sequence_to_predict[0] - reconstructed_sequence[0], 2)
            mse_per_feature = np.mean(sq_err_elems, axis=0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for i, fnk in enumerate(self.metric_columns_ordered):
                    try:
                        cfm = mse_per_feature[i]
                        PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(
                            feature_name=fnk).set(cfm)
                        if mse > threshold*0.5 or cfm > threshold*0.5:
                            logging.info(f"Mistake'{fnk}': {cfm:.6f}")
                    except Exception as e:
                        logging.error(f"Oral error metrics for'{fnk}': {e}")
            is_anomaly = mse > threshold
            if is_anomaly:
                logging.warning(
                    f"!!! Anomaly!! MSE: {mse:.6f} > Threshold: 6f}")
                if PROM_IS_ANOMALY_DETECTED:
                    PROM_IS_ANOMALY_DETECTED.set(1)
                if PROM_TOTAL_ANOMALIES_COUNT:
                    PROM_TOTAL_ANOMALIES_COUNT.inc()
                fer = ["Errors by signs (anomaly):"]
                for i, fnk in enumerate(self.metric_columns_ordered):
                    fer.append(f"  - '{fnk}': {mse_per_feature[i]:.6f}")
                logging.warning("\n".join(fer))
            else:
                logging.info(
                    f"Norma. MSE: {mse:.6f} <= Threshold:.6f}")
                if PROM_IS_ANOMALY_DETECTED:
                    PROM_IS_ANOMALY_DETECTED.set(0)
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            if PROM_LATEST_RECONSTRUCTION_ERROR_MSE:
                PROM_LATEST_RECONSTRUCTION_ERROR_MSE.set(-1)
            if PROM_IS_ANOMALY_DETECTED:
                PROM_IS_ANOMALY_DETECTED.set(0)
            if PROM_FEATURE_RECONSTRUCTION_ERROR_MSE:
                for fnk in self.metric_columns_ordered:
                    PROM_FEATURE_RECONSTRUCTION_ERROR_MSE.labels(
                        feature_name=fnk).set(-1)

    def run_detection_cycle(self):
        logging.info("Beginning of the cycle.")
        current_window_df = self._fetch_data_window()
        # Use corrected names for global variable metrics
        if PROM_DATA_POINTS_IN_CURRENT_WINDOW:
            PROM_DATA_POINTS_IN_CURRENT_WINDOW.set(
                len(current_window_df) if current_window_df is not None else 0)
        if current_window_df is None or current_window_df.empty:
            logging.warning("No window data. Pass.")
            self._process_model_output(
                None, np.array([]), self.anomaly_threshold)
            return
        sequence_to_predict = self._preprocess_and_create_sequence(
            current_window_df.copy())
        if sequence_to_predict is None:
            logging.warning("It was not possible to pre-process the data. Pass.")
            self._process_model_output(
                None, np.array([]), self.anomaly_threshold)
            return
        logging.info("--- Model processing ---")
        self._process_model_output(
            self.model_a, sequence_to_predict, self.anomaly_threshold)
        if PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS:
            PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS.set_to_current_time()
        logging.info("Cycle complete.")

    def start_server_and_loop(self):
        if not self.scaler or not self.model_a:
            logging.error("No skater or model loaded. Launch impossible.")
            exit(1)
        try:
            start_http_server(self.exporter_port)
            logging.info(f"Prometheus exporter at the port {self.exporter_port}")
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:
                logging.error(f"Port {self.exporter_port} is busy.")
            else:
                logging.error(f"OSError exporter: {e}", exc_info=True)
            exit(1)
        except Exception as e:
            logging.error(f"Error exporter: {e}", exc_info=True)
            exit(1)
        while True:
            try:
                self.run_detection_cycle()
            except Exception as e:
                logging.error(f"Critical error in the cycle: {e}", exc_info=True)
            logging.info(f"Waiting {self.query_interval}c. .")
            time.sleep(self.query_interval)


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    PROM_LATEST_RECONSTRUCTION_ERROR_MSE = None
    PROM_IS_ANOMALY_DETECTED = None
    PROM_TOTAL_ANOMALIES_COUNT = None
    PROM_FEATURE_RECONSTRUCTION_ERROR_MSE = None
    PROM_LAST_SUCCESSFUL_RUN_TIMESTAMP_SECONDS = None
    PROM_DATA_POINTS_IN_CURRENT_WINDOW = None
    detector = RealtimeAnomalyDetector(config_path=config_file)
    detector.start_server_and_loop()