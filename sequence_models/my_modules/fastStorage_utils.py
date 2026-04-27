import csv
from datetime import datetime
import glob
import os
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from my_modules.printable_utils import logs

def preprocess_data_from_fastStorage(
        input_pattern, 
        features, 
        SAMPLE_SIZE_AVERAGE,
        scaler,
        normalize_scaler,
        target_col = 'cpu_avg'
    ):
    '''
    Change Data to Correct Format
    
    Parameters
    -----------
    - input_pattern: glob pattern for files
    - features: list of feature names
    - SAMPLE_SIZE_AVERAGE: number of raw samples per averaged step
    - scaler: optional feature scaler (0–1)
    - normalize_scaler: optional target normalizer (0–100)
    - target_col: name of target column

    Returns
    --------
    - data: np.ndarray of scaled flatten dataset
    - normalize_scaler
    - scaler
    '''

    dfs = []

    files = glob.glob(input_pattern)
    if not files:
        logs(f"Warning: No files matched pattern: {input_pattern}")
        return np.array([]), None, None
    logs(f"Found {len(files)} files.")

    for fname in files:
        processed_rows = []
        buffers = {"timestamp": [], "cpu": [], "mem_percent": []}

        with open(fname, 'r', encoding='utf-8') as infh:
            reader = csv.reader(infh, delimiter=';')
            try:
                next(reader)  # Skip header
            except StopIteration:
                logs(f"Warning: File {fname} is empty.")
                continue

            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    timestamp = int(row[0])
                    cpu_usage = float(row[4])
                    mem_capacity = float(row[5])
                    mem_usage = float(row[6])
                    mem_usage_percent = (mem_usage / mem_capacity * 100.0) if mem_capacity > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    continue

                buffers["timestamp"].append(timestamp)
                buffers["cpu"].append(cpu_usage)
                buffers["mem_percent"].append(mem_usage_percent)

                if len(buffers["timestamp"]) >= SAMPLE_SIZE_AVERAGE:
                    timestamp_avg = mean(buffers["timestamp"])
                    dt = datetime.fromtimestamp(timestamp_avg)

                    hour = dt.hour
                    minute = dt.minute
                    day_of_week = dt.weekday()
                    month = dt.month

                    cpu_avg = mean(buffers["cpu"])
                    mem_avg = mean(buffers["mem_percent"])

                    processed_rows.append([
                        timestamp_avg,
                        hour, minute, day_of_week, month,
                        cpu_avg, mem_avg
                    ])

                    for key in buffers:
                        buffers[key].clear()

        if processed_rows:
            df = pd.DataFrame(processed_rows, columns=features)
            df['cpu_id'] = os.path.basename(fname)
            dfs.append(df)
        else:
            logs(f"File {fname} produced no processed rows.")

    if not dfs:
        logs(f"No valid data processed from any files.")
        return np.array([]), None, None

    # Combine all processed DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='hour')

    # Normalize each CPU group separately
    scalers = {}
    scaled_dfs = []
    for cpu_id, group in combined_df.groupby('cpu_id'):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(group[features])
        scaled_df = pd.DataFrame(scaled, columns=features)
        scaled_df['cpu_id'] = cpu_id
        scalers[cpu_id] = scaler
        scaled_dfs.append(scaled_df)

    data = pd.concat(scaled_dfs, ignore_index=True)

    # Flatten CPU average series for LSTM
    time_series_data = data[target_col].values.flatten()

    # Normalize 0–100 for interpretability
    # if normalize_scaler is None:
    #     normalize_scaler = MinMaxScaler(feature_range=(0, 100))
    # normalized_data = normalize_scaler.fit_transform(time_series_data.reshape(-1, 1))
    # scaled_data = scaler.fit_transform(normalized_data)

    # Scale 0–1 for LSTM
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data)

    return scaled_data.flatten(), normalize_scaler, scaler



def preprocess_data_from_fastStorage_multi_input(
        input_pattern, 
        input_features,
        target_features,
        SAMPLE_SIZE_AVERAGE,
        scaler=None,
        normalize_scaler=None
    ):
    """
    Preprocess FastStorage dataset for multi-input / multi-output models (LSTM/Hybrid).

    Parameters
    -----------
    - input_pattern: glob pattern for files
    - input_features: list of feature names used as inputs
    - target_features: list of target names (e.g., ["cpu_avg"])
    - SAMPLE_SIZE_AVERAGE: number of raw samples per averaged step
    - scaler: optional feature scaler (0–1)
    - normalize_scaler: optional target normalizer (0–100)

    Returns
    --------
    - X: np.ndarray of scaled input features
    - y: np.ndarray of scaled target features
    - normalize_scaler
    - scaler
    """
    dfs = []
    files = glob.glob(input_pattern)
    if not files:
        logs(f"Warning: No files matched pattern: {input_pattern}")
        return np.array([]), np.array([]), normalize_scaler, scaler

    logs(f"Found {len(files)} files.")

    for fname in files:
        processed_rows = []
        buffers = {"timestamp": [], "cpu": [], "mem_percent": []}

        with open(fname, 'r', encoding='utf-8') as infh:
            reader = csv.reader(infh, delimiter=';')
            try:
                next(reader)  # Skip header
            except StopIteration:
                logs(f"Warning: File {fname} is empty.")
                continue

            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    timestamp = int(row[0])
                    cpu_usage = float(row[4])
                    mem_capacity = float(row[5])
                    mem_usage = float(row[6])
                    mem_usage_percent = (mem_usage / mem_capacity * 100.0) if mem_capacity > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    continue

                buffers["timestamp"].append(timestamp)
                buffers["cpu"].append(cpu_usage)
                buffers["mem_percent"].append(mem_usage_percent)

                if len(buffers["timestamp"]) >= SAMPLE_SIZE_AVERAGE:
                    timestamp_avg = mean(buffers["timestamp"])
                    dt = datetime.fromtimestamp(timestamp_avg)

                    hour = dt.hour
                    minute = dt.minute
                    day_of_week = dt.weekday()
                    month = dt.month

                    cpu_avg = mean(buffers["cpu"])
                    mem_avg = mean(buffers["mem_percent"])

                    processed_rows.append([
                        timestamp_avg,
                        hour, minute, day_of_week, month,
                        cpu_avg, mem_avg
                    ])

                    for key in buffers:
                        buffers[key].clear()

        if processed_rows:
            columns = [
                "timestamp", "hour", "minute", "day_of_week", "month",
                "cpu_avg", "mem_avg"
            ]
            df = pd.DataFrame(processed_rows, columns=columns)
            df["cpu_id"] = os.path.basename(fname)
            dfs.append(df)
        else:
            logs(f"File {fname} produced no processed rows.")

    if not dfs:
        logs(f"No valid data processed from any files.")
        return np.array([]), np.array([]), normalize_scaler, scaler

    # Combine and sort
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=["cpu_id", "hour"]).reset_index(drop=True)

    # Scale inputs (0–1)
    scaled_dfs = []
    scalers = {}
    for cpu_id, group in combined_df.groupby("cpu_id"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(group[input_features])
        scaled_df = pd.DataFrame(scaled, columns=input_features)
        scaled_df["cpu_id"] = cpu_id
        scalers[cpu_id] = scaler
        scaled_dfs.append(scaled_df)

    data_scaled = pd.concat(scaled_dfs, ignore_index=True)

    # Extract target(s) -> Ensure target_features is a list and make 2D array
    if isinstance(target_features, str):
        target_features = [target_features]

    y_raw = combined_df[target_features].values

    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)


    # Normalize targets 0–100
    if normalize_scaler is None:
        normalize_scaler = MinMaxScaler(feature_range=(0, 100))
    y_normalized = normalize_scaler.fit_transform(y_raw)

    # Scale targets 0–1
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y_normalized)

    # Final input matrix
    X = data_scaled[input_features].values
    y = y_scaled

    logs(f"Preprocessed multi-input data: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, normalize_scaler, scaler
