import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path
from diskcache import Cache

# --- Global variables ---
CONFIG = {}
CACHE: Cache | None = None

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


def query_prometheus_range(prometheus_url: str, query: str, start_time: datetime, end_time: datetime, step: str) -> pd.DataFrame:
    """Requests data from Prometheus, using cache to speed up repeated requests."""
    if CACHE is not None:
        cache_key = (prometheus_url, query, start_time.isoformat(), end_time.isoformat(), step)
        cached_result = CACHE.get(cache_key)
        if cached_result is not None:
            # Make sure we download a copy to avoid problems with changeability.
            return cached_result.copy()

    api_url = f"{prometheus_url}/api/v1/query_range"
    params = {'query': query, 'start': start_time.timestamp(), 'end': end_time.timestamp(), 'step': step}
    print(f"  CACHE MISS: Prometheus: {query[:70]} ({start_time.strftime}'%Y-%m-%d %H:%M')} -> {end_time.strftime('%Y-%m-%d %H:%M')})")
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"    -> Error when requesting Prometheus: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"    -> Error decoding JSON response from Prometheus: {e}\n Response text: {response.text}")
        return pd.DataFrame()

    result_df = pd.DataFrame()
    if data['status'] == 'success':
        all_series_data = []
        for result in data['data']['result']:
            metric_labels = result.get('metric', {})
            values = result.get('values', [])
            if not values: continue
            
            df_series = pd.DataFrame(values, columns=['timestamp', 'value'])
            df_series['timestamp'] = pd.to_datetime(df_series['timestamp'], unit='s')
            df_series['value'] = pd.to_numeric(df_series['value'], errors='coerce')
            df_series = df_series.set_index('timestamp')
            all_series_data.append(df_series)

        if not all_series_data: pass
        elif len(all_series_data) == 1:
            result_df = all_series_data[0]
        else:
            print(f"    -> Warning: request'{query}' The time series has been returned. Return the first.")
            result_df = all_series_data[0]
    else:
        print(f"    -> Error in the status of the answer Prometheus: {data.get()'errorType')}, {data.get('error')}")

    if CACHE is not None and not result_df.empty:
        CACHE.set(cache_key, result_df)

    return result_df


def collect_training_data(prometheus_url: str, queries_dict: dict, start_time: datetime, end_time: datetime, step: str, chunk_hours: int) -> pd.DataFrame:
    """
    Collects data by breaking up a large time range into smaller chunks for efficient caching.
    """
    all_chunks_data = []
    current_start = start_time
    print(f"Break the period with {start_time.strftime()'%Y-%m-%d %H:%M')} by {end_time.strftime()'%Y-%m-%d %H:%M')} {chunk_hours} hour(a).")

    while current_start < end_time:
        current_end = current_start + timedelta(hours=chunk_hours)
        if current_end > end_time:
            current_end = end_time

        print(f"\n-- Data collection for chunk: {current_start.strftime()'%Y-%m-%d %H:%M')} -> {current_end.strftime('%Y-%m-%d %H:%M')}")
        
        metrics_for_chunk = []
        for custom_name, query_string in queries_dict.items():
            df_metric = query_prometheus_range(prometheus_url, query_string, current_start, current_end, step)
            if not df_metric.empty:
                df_metric = df_metric.rename(columns={'value': custom_name})
                metrics_for_chunk.append(df_metric)

        if metrics_for_chunk:
            chunk_df = pd.concat(metrics_for_chunk, axis=1, join='outer')
            all_chunks_data.append(chunk_df)
        
        current_start = current_end
        
    if not all_chunks_data:
        print("\nNo data was collected for the entire period.")
        return pd.DataFrame()

    print("\nCombining data from all chunks. . .")
    final_df = pd.concat(all_chunks_data, axis=0)

    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    expected_column_names = list(queries_dict.keys())
    for col_name in expected_column_names:
        if col_name not in final_df.columns:
            final_df[col_name] = pd.NA 

    final_df = final_df.sort_index()

    return final_df

# --- Basic block ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    CONFIG_FILE_PATH = BASE_DIR / "config.yaml"
    CONFIG = load_config(CONFIG_FILE_PATH)

    artifacts_path_str = CONFIG.get('artifacts_dir', 'artifacts')
    artifacts_dir = BASE_DIR / artifacts_path_str
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory for artifacts: {artifacts_dir}")

    cache_dir = artifacts_dir / "prometheus_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        CACHE = Cache(str(cache_dir))
        print(f"Prometheus query caching is enabled. Cache directory: {cache_dir}")
    except Exception as e:
        print(f"The cache directory was not created {cache_dir}: {e}. Caching's off.")
        CACHE = None

    PROMETHEUS_URL = CONFIG.get('prometheus_url')
    QUERIES = CONFIG.get('queries')
    DATA_SETTINGS = CONFIG.get('data_settings', {})

    if not PROMETHEUS_URL or not QUERIES:
        print("Mistake:'prometheus_url' or'queries' Not found in the configuration file.")
        exit(1)

    STEP = DATA_SETTINGS.get('step', '30s')
    CHUNK_HOURS = DATA_SETTINGS.get('cache_chunk_hours', 1)
    PARQUET_FILENAME = DATA_SETTINGS.get('output_filename', 'prometheus_metrics_data.parquet')
    output_file_path = artifacts_dir / PARQUET_FILENAME

    print(f"\nData collection from Prometheus: {PROMETHEUS_URL}")
    print(f"Step: {STEP} | Chank size for caching: {CHUNK_HOURS} hour(a)")
    print(f"File to save: {output_file_path}")
    print("-" * 30)

    training_data_list = []
    collection_periods = DATA_SETTINGS.get('collection_periods_iso')

    if collection_periods and isinstance(collection_periods, list):
        print("Multi-period configuration discovered'collection_periods_iso'.")
        for i, period in enumerate(collection_periods):
            try:
                start_time = datetime.fromisoformat(period['start'])
                end_time = datetime.fromisoformat(period['end'])
                if start_time >= end_time:
                    print(f"Error in the period {i+1}:'start' ({start_time}) should have been sooner'end' ({end_time}). Period skip.")
                    continue
                
                print(f"\n--- Period treatment {i+1}/{len(collection_periods)} ---")
                period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP, CHUNK_HOURS)
                if not period_df.empty:
                    training_data_list.append(period_df)

            except (KeyError, ValueError) as e:
                print(f"Error in period configuration {i+1}: {e} Period skip.")
                continue
    else:
        print("The configuration'collection_periods_iso' not found, used'collection_period_hours' or'start_time_iso'/'end_time_iso'.")
        collection_period_hours = DATA_SETTINGS.get('collection_period_hours')
        start_time_iso = DATA_SETTINGS.get('start_time_iso')
        end_time_iso = DATA_SETTINGS.get('end_time_iso')
        
        start_time, end_time = None, None
        
        if start_time_iso and end_time_iso and (not collection_period_hours or collection_period_hours == 0):
            try:
                start_time = datetime.fromisoformat(start_time_iso)
                end_time = datetime.fromisoformat(end_time_iso)
                if start_time >= end_time:
                    print("Mistake:'start_time_iso' should have been sooner'end_time_iso'.")
                    exit(1)
            except ValueError:
                print("Mistake:'start_time_iso' or'end_time_iso' They have the wrong format. Use YYYY-MM-DDTHH:MM:SS.")
                exit(1)
        elif collection_period_hours and collection_period_hours > 0:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=collection_period_hours)
        else:
            print("Mistake: Incorrect timing settings. Indicate'collection_periods_iso', 'collection_period_hours' > 0 or'start_time_iso' and'end_time_iso'.")
            exit(1)
            
        if start_time and end_time:
            print(f"\n--- Processing a single period ---")
            single_period_df = collect_training_data(PROMETHEUS_URL, QUERIES, start_time, end_time, STEP, CHUNK_HOURS)
            if not single_period_df.empty:
                training_data_list.append(single_period_df)

    if training_data_list:
        final_training_data = pd.concat(training_data_list)
        final_training_data = final_training_data.sort_index()
        final_training_data = final_training_data[~final_training_data.index.duplicated(keep='first')]

        final_training_data['day_of_week'] = final_training_data.index.dayofweek.astype(int)
        final_training_data['hour_of_day'] = final_training_data.index.hour.astype(int)

        print("\n--- Data collected (first 5 and last 5 lines)")
        print(final_training_data.head())
        print("...")
        print(final_training_data.tail())
        print(f"\nDataFrame size: {final_training_data.shape}")

        try:
            final_training_data.to_parquet(output_file_path, engine='pyarrow', index=True)
            print(f"\nThe data is successfully stored in {output_file_path}")
        except Exception as e:
            print(f"\nError in saving data in Parquet: {e}")
            print("Make sure that the library'pyarrow' (or'fastparquet') set.")
    else:
        print("\nNo data were collected for these periods.")

    if CACHE is not None:
        CACHE.close()