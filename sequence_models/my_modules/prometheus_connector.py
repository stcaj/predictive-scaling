import requests
import pandas as pd
import numpy as np

from typing import Dict, Any
from typing import Tuple
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from my_modules.printable_utils import logs

def get_query(query: str, start_ts: int, end_ts: int, step: int):
    '''
    returns right query by input parameters
    '''
    return {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": step
    }

def load_prometheus_data(
        deployment_name: str,
        PROMETHEUS_API_URL: str,
        start_ts: int,
        end_ts: int,
        step: int,
        namespace = "default"
    ):
    '''
    Load revelant data from Prometheus
    
    Parameters:
    - deployment_name: name of the deployment / service
    - PROMETHEUS_API_URL: api of prometheus
    - start_ts: time from which data to load
    - end_ts: time till which data to load
    - step: step between values
    - namespace: namespace of the deployment

    '''
    # Total CPU usage across all pods
    cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{deployment_name}-.*"}}[{2*step}s]))'
    # cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{deployment_name}-.*"}}[5m]))'
    cpu_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query_range", params=get_query(cpu_query, start_ts, end_ts, step))
    cpu_data = cpu_resp.json()

    usage_series = [(int(ts), float(usage)) for ts, usage in cpu_data["data"]["result"][0]["values"]] \
        if cpu_data["data"]["result"] else []

    # CPU limits
    # cpu_limit_query = f'sum(kube_pod_container_resource_limits{{pod=~"{deployment_name}-.*", resource="cpu", namespace="{namespace}"}})'
    cpu_limit_query = f'sum(kube_pod_container_resource_requests{{pod=~"{deployment_name}-.*", resource="cpu", namespace="{namespace}"}})'
    cpu_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query", params={"query": cpu_limit_query})
    cpu_data = cpu_resp.json()
    total_cpu_limit = float(cpu_data["data"]["result"][0]["value"][1]) if cpu_data["data"]["result"] else 0

    # Total Memory usage across all pods
    mem_query = f'sum(container_memory_working_set_bytes{{pod=~"{deployment_name}-.*"}})'
    mem_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query_range", params=get_query(mem_query, start_ts, end_ts, step))
    mem_data = mem_resp.json()
    memory_series = mem_data["data"]["result"][0]["values"] if mem_data["data"]["result"] else []

    # Build dict for memory values
    memory_dict = {int(ts): float(val) for ts, val in memory_series}

    # Memory limits
    # mem_limit_query = f'sum(kube_pod_container_resource_limits{{pod=~"{deployment_name}-.*", resource="memory", namespace="{namespace}"}})'
    mem_limit_query = f'sum(kube_pod_container_resource_requests{{pod=~"{deployment_name}-.*", resource="memory", namespace="{namespace}"}})'
    mem_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query", params={"query": mem_limit_query})
    mem_data = mem_resp.json()
    total_mem_limit = float(mem_data["data"]["result"][0]["value"][1]) if mem_data["data"]["result"] else 0

    # Total replicas for the deployment
    replicas_query = f'kube_deployment_spec_replicas{{deployment="{deployment_name}", namespace="{namespace}"}}'
    replicas_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query", params={"query": replicas_query})
    replicas_data = replicas_resp.json()
    desired_replicas = int(replicas_data["data"]["result"][0]["value"][1]) if replicas_data["data"]["result"] else None

    available_query = f'kube_deployment_status_replicas_available{{deployment="{deployment_name}", namespace="{namespace}"}}'
    available_resp = requests.get(f"{PROMETHEUS_API_URL}/api/v1/query", params={"query": available_query})
    available_data = available_resp.json()
    available_replicas = int(available_data["data"]["result"][0]["value"][1]) if available_data["data"]["result"] else None    

    return usage_series, memory_dict, total_cpu_limit, total_mem_limit, desired_replicas, available_replicas

def get_pod_data(
        deployment_name: str, 
        start_time: datetime, 
        end_time: datetime,
        step: int,
        PROMETHEUS_API_URL: str, 
        namespace: str = "default"
    ) -> pd.DataFrame:
    """
    Get data from Prometheus for a deployment (all replicas).
    CPU and memory values are matched by timestamp (not index).
    """
    usage_series, memory_dict, total_cpu_limit, total_mem_limit, desired_replicas, available_replicas = load_prometheus_data(
        deployment_name=deployment_name,
        PROMETHEUS_API_URL=PROMETHEUS_API_URL,
        start_ts=int(start_time.timestamp()),
        end_ts=int(end_time.timestamp()),
        step=step,
        namespace=namespace
    )

    # Build rows
    rows = []
    for ts, cpu_val in usage_series:
        dt = datetime.fromtimestamp(ts)
        mem_val = memory_dict.get(ts, None)

        if total_cpu_limit > 0 and cpu_val is not None:
            cpu_perc = (cpu_val / total_cpu_limit) * 100
        else:
            cpu_perc = 0.0

        if total_mem_limit > 0 and mem_val is not None:
            mem_perc = (mem_val / total_mem_limit) * 100
        else:
            mem_perc = 0.0

        row = {
            "timestamp": int(ts),
            "hour": dt.hour,
            "minute": dt.minute,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "deployment": deployment_name,
            "cpu": cpu_val,
            "cpu_percent": cpu_perc,
            "memory_bytes": mem_val,
            "memory_percent": mem_perc,
            "desired_replicas": desired_replicas,
            "available_replicas": available_replicas
        }
        rows.append(row)


    data = pd.DataFrame(rows)
    data = data.fillna(0.0)
    return data

def load_microservice_data(
        deployments: list, 
        PROMETHEUS_URL: str, 
        start_time: int, 
        end_time: int,
        step: int
    ) -> pd.DataFrame:
    '''
    Make a dataframe with all data from microservices of online butique microservice app on kubernetes using API of prometheus

    Parameters:
    - deployments: list of deployment names
    - PROMETHEUS_URL: url and port of prometheus
    - start_time: time from which the data are loaded
    - end_time: time till which the data are loaded
    '''

    # make dataframe
    all_dfs = []
    for deploy in deployments:
        df = get_pod_data(
            deployment_name=deploy, 
            start_time=start_time,
            step=step,
            end_time=end_time,
            PROMETHEUS_API_URL=PROMETHEUS_URL
        )
        if not df.empty:
            all_dfs.append(df)

    logs(f"Finished classifying data.")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()



def preprocess_data(
        dataframe: pd.DataFrame,
        features: list,
        deployments,
        target_col=None,
        target_features=None,
        scaler = None,
        normalize_scaler = None,
    ):
    '''
    Preproces datframe given to a format suitable for machine learning

    Parameters:
    - dataframe: dataframe loaded from prometheus
    - features: features loded from prometheus -> inside config.yaml
    - target_col: the target col that the user looks for -> this case cpu (cpu usage in percent)
    - scaler: used scaler
    - normalize_scaler: used normalizer
    '''
    # if dataframe empty
    if dataframe is None or dataframe.empty:
        logs("No data provided to preprocess_data()")
        return None, normalize_scaler, scaler

    combined_df = dataframe.copy()
    if "cpu_id" not in combined_df.columns:
        combined_df["cpu_id"] = "provided_df"

    combined_df = combined_df.sort_values(by=["deployment_id", "timestamp"]).reset_index(drop=True)

    # Normalize each CPU group separately
    scaled_dfs = []
    scalers = {}
    for cpu_id, group in combined_df.groupby("cpu_id"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(group[features])
        scaled_df = pd.DataFrame(scaled, columns=features)
        scaled_df["cpu_id"] = cpu_id
        scalers[cpu_id] = scaler
        scaled_dfs.append(scaled_df)

    data = pd.concat(scaled_dfs, ignore_index=True)

    # choose targets
    if target_col is not None and target_features is not None:
        logs("Both target_col and target_features provided — using target_features.")
        time_series_data = data[target_features].values
    elif target_col is not None:
        time_series_data = data[[target_col]].values
    elif target_features is not None:
        time_series_data = data[target_features].values
    else:
        raise ValueError("You must provide either target_col or target_features.")


    if normalize_scaler is None:
        normalize_scaler = MinMaxScaler(feature_range=(0, 100))
    normalized_data = normalize_scaler.fit_transform(time_series_data.reshape(-1, 1))

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(normalized_data)

    return scaled_data.flatten(), normalize_scaler, scaler

# def preprocess_data_multi(
#         dataframe: pd.DataFrame,
#         features: list,
#         deployments,
#         target_col=None,
#         target_features=None,
#         scaler=None,
#         normalize_scaler=None,
#     ):
#     """
#     Preprocess dataframe into scaled input (X) and target (y) arrays
#     suitable for multi-feature RNN/MLP models.

#     Parameters:
#     - dataframe: raw data from Prometheus
#     - features: list of input features used for X
#     - deployments: list of deployment names for label encoding
#     - target_col: single target name (e.g., 'cpu_percent')
#     - target_features: list of target column names (multi-output)
#     - scaler: existing feature scaler (reuse if not None)
#     - normalize_scaler: existing target scaler (reuse if not None)

#     Returns:
#     - X_scaled: np.ndarray (samples, num_features)
#     - y_scaled: np.ndarray (samples, num_targets)
#     - normalize_scaler
#     - scaler
#     """
#     if dataframe is None or dataframe.empty:
#         logs("No data provided to preprocess_data()")
#         return None, None, normalize_scaler, scaler

#     # Encode deployments
#     le = LabelEncoder()
#     le.fit(deployments)
#     dataframe["deployment_id"] = le.transform(dataframe["deployment"])

#     # Sort by deployment + timestamp (ensure chronological order)
#     df = dataframe.copy()
#     df = df.sort_values(by=["deployment_id", "timestamp"]).reset_index(drop=True)

#     # Scale input features
#     if scaler is None:
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         X_scaled = scaler.fit_transform(df[features])
#     else:
#         X_scaled = scaler.transform(df[features])

#     # Select targets
#     if target_col is not None and target_features is not None:
#         logs("Both target_col and target_features provided — using target_features.")
#         y_raw = df[target_features].values
#     elif target_col is not None:
#         y_raw = df[[target_col]].values
#     elif target_features is not None:
#         y_raw = df[target_features].values
#     else:
#         raise ValueError("You must provide either target_col or target_features.")

#     # Normalize targets
#     if normalize_scaler is None:
#         normalize_scaler = MinMaxScaler(feature_range=(0, 1))
#         y_scaled = normalize_scaler.fit_transform(y_raw)
#     else:
#         y_scaled = normalize_scaler.transform(y_raw)

#     return X_scaled, y_scaled, normalize_scaler, scaler

def preprocess_data_multi(
        dataframe: pd.DataFrame,
        features: list,
        target_col=None,
        target_features=None,
        scaler=None,
        normalize_scaler=None,
    ):
    if dataframe is None or dataframe.empty:
        logs("No data provided to preprocess_data_multi()")
        return None, None, normalize_scaler, scaler

    df = dataframe.copy()

    if "timestamp" not in df.columns:
        raise KeyError("Missing required column: 'timestamp'")

    df = df.sort_values(by=["timestamp"]).reset_index(drop=True)

    # --- cyclic feature transformation ---
    features_expanded = []

    for col in features:
        if col == "hour":
            if "hour" not in df.columns:
                raise KeyError("Missing feature column: 'hour'")
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
            features_expanded.extend(["hour_sin", "hour_cos"])

        elif col == "day_of_week":
            if "day_of_week" not in df.columns:
                raise KeyError("Missing feature column: 'day_of_week'")
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
            features_expanded.extend(["dow_sin", "dow_cos"])

        else:
            if col not in df.columns:
                raise KeyError(f"Missing feature column: '{col}'")
            features_expanded.append(col)

    # --- split features into scaled and unscaled ---
    cyclic_features = [c for c in features_expanded if c in {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}]
    scaled_features = [c for c in features_expanded if c not in cyclic_features]

    # scale only non-cyclic features
    if scaled_features:
        X_scaled_part_raw = df[scaled_features].values
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled_part = scaler.fit_transform(X_scaled_part_raw)
        else:
            X_scaled_part = scaler.transform(X_scaled_part_raw)
    else:
        X_scaled_part = np.empty((len(df), 0), dtype=np.float32)

    # keep cyclic features as-is
    if cyclic_features:
        X_cyclic_part = df[cyclic_features].values.astype(np.float32)
    else:
        X_cyclic_part = np.empty((len(df), 0), dtype=np.float32)

    # preserve original feature order after expansion
    feature_arrays = {}
    scaled_idx = 0
    cyclic_idx = 0

    for col in features_expanded:
        if col in cyclic_features:
            feature_arrays[col] = X_cyclic_part[:, cyclic_idx:cyclic_idx + 1]
            cyclic_idx += 1
        else:
            feature_arrays[col] = X_scaled_part[:, scaled_idx:scaled_idx + 1]
            scaled_idx += 1

    X_final = np.hstack([feature_arrays[col] for col in features_expanded]).astype(np.float32)

    # --- targets ---
    if target_col is not None and target_features is not None:
        logs("Both target_col and target_features provided — using target_features.")
        y_raw = df[target_features].values
    elif target_col is not None:
        if target_col not in df.columns:
            raise KeyError(f"Missing target column: '{target_col}'")
        y_raw = df[[target_col]].values
    elif target_features is not None:
        missing_targets = [c for c in target_features if c not in df.columns]
        if missing_targets:
            raise KeyError(f"Missing target columns: {missing_targets}")
        y_raw = df[target_features].values
    else:
        raise ValueError("You must provide either target_col or target_features.")

    if normalize_scaler is None:
        normalize_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = normalize_scaler.fit_transform(y_raw)
    else:
        y_scaled = normalize_scaler.transform(y_raw)

    logs(f"Original features: {features}")
    logs(f"Expanded features: {features_expanded}")
    logs(f"Scaled features: {scaled_features}")
    logs(f"Unscaled cyclic features: {cyclic_features}")
    logs(f"X_final shape: {X_final.shape}")
    logs(f"y_scaled shape: {y_scaled.shape}")

    return (
        np.asarray(X_final, dtype=np.float32),
        np.asarray(y_scaled, dtype=np.float32),
        normalize_scaler,
        scaler,
    )


def load_prometheus_multi(
        metadata,
        features,
        PROMETHEUS_URL,
        deployments,
        start_time: int, 
        end_time: int,
        scaler = None,
        normalizer = None,
        target = "cpu_percent",
        target_features = None
    ) -> Tuple:
    """
    Load -> Preprocess -> Split Prometheus data for ML using class labels.

    Parameters:
    - scaler: scaler loaded
    - normalzier: normalizer loaded
    - metadata: metadata loaded from config.yaml
    - features: fetaures loaded from prometheus
    - PROMETHEUS_URL: url of prometheus
    - deployments: deployment (microservices) of application
    - start_time: time from data are loaded
    - end_time: time til data are laoded
    - target: target collumn that wil be taken from dataset
    - target_features: target collumns that will be taken from dataset

    Returns:
    - data: preprocessed data
    - scaler: scaler used
    - normalizer: normalzier used
    """


    # Load raw data
    data = load_microservice_data(
        deployments=deployments, 
        PROMETHEUS_URL=PROMETHEUS_URL,
        start_time=start_time, 
        end_time=end_time,
        step=metadata["data_param"]["step_seconds"]
    )
    
    logs(f"Finished loading data.")
    
    # Preprocess data
    X_scaled, y_scaled, normalize, scaler = preprocess_data_multi(
        dataframe=data,
        features=features,
        deployments=deployments,
        target_col=target,
        target_features=target_features,
        scaler=scaler,
        normalize_scaler=normalizer
    )
    return X_scaled, y_scaled, normalize, scaler

def load_prometheus(
        metadata,
        features,
        PROMETHEUS_URL,
        deployments,
        start_time: int, 
        end_time: int,
        scaler = None,
        normalizer = None,
        target = "cpu_percent",
        target_features = None
    ) -> Tuple:
    """
    Load -> Preprocess -> Split Prometheus data for ML using class labels.

    Parameters:
    - scaler: scaler loaded
    - normalzier: normalizer loaded
    - metadata: metadata loaded from config.yaml
    - features: fetaures loaded from prometheus
    - PROMETHEUS_URL: url of prometheus
    - deployments: deployment (microservices) of application
    - start_time: time from data are loaded
    - end_time: time til data are laoded
    - target: target collumn that wil be taken from dataset
    - target_features: target collumns that will be taken from dataset

    Returns:
    - data: preprocessed data
    - scaler: scaler used
    - normalizer: normalzier used
    """


    # Load raw data
    data = load_microservice_data(
        deployments=deployments, 
        PROMETHEUS_URL=PROMETHEUS_URL,
        start_time=start_time, 
        end_time=end_time,
        step=metadata["data_param"]["step_seconds"]
    )
    
    logs(f"Finished loading data.")
    
    # Preprocess data
    data, normalize, scaler = preprocess_data(
        dataframe=data,
        features=features,
        deployments=deployments,
        target_col=target,
        target_features=target_features,
        scaler=scaler,
        normalize_scaler=normalizer
    )
    return data, normalize, scaler

def load_data_for_prediction(
        metadata,
        PROMETHEUS_URL,
        deployments,
        start_time: int,
        end_time: int,
        seq_length: int,
        feature_columns=None,
    ) -> Dict[str, Any]:
    
    if feature_columns is None or len(feature_columns) == 0:
        feature_columns = ["cpu_percent"]

    # Load raw data from Prometheus
    data = load_microservice_data(
        deployments=deployments,
        PROMETHEUS_URL=PROMETHEUS_URL,
        start_time=start_time,
        end_time=end_time,
        step=metadata["data_param"]["step_seconds"]
    )

    if data.empty:
        logs("No data available for prediction.")
        return {}

    result = {}

    for deploy in deployments:
        deploy_data = (
            data[data["deployment"] == deploy]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if deploy_data.empty:
            logs(f"No data for deployment '{deploy}', skipping.")
            continue

        missing_cols = [col for col in feature_columns if col not in deploy_data.columns]
        if missing_cols:
            logs(f"Missing feature columns for deployment '{deploy}': {missing_cols}")
            continue

        raw_data = deploy_data[feature_columns].values.astype(np.float32)

        sequences = []
        timestamps = []

        if len(raw_data) < seq_length:
            pad_len = seq_length - len(raw_data)
            padded = np.vstack([
                np.zeros((pad_len, raw_data.shape[1]), dtype=np.float32),
                raw_data
            ])
            sequences.append(padded)
            timestamps.append(int(deploy_data["timestamp"].iloc[-1]))
            logs(f"Padded deployment '{deploy}' with {pad_len} zero rows.")
        else:
            for i in range(len(raw_data) - seq_length + 1):
                seq = raw_data[i:i + seq_length]
                sequences.append(seq)
                timestamps.append(int(deploy_data["timestamp"].iloc[i + seq_length - 1]))

        result[deploy] = {
            "X_raw": np.asarray(sequences, dtype=np.float32),
            "timestamps": timestamps,
            "feature_columns": feature_columns
        }

        logs(
            f"[PRED] Deployment={deploy}, "
            f"X_raw shape={result[deploy]['X_raw'].shape}, "
            f"num_timestamps={len(timestamps)}"
        )

    return result

# def load_data_for_prediction(
#         metadata,
#         PROMETHEUS_URL,
#         deployments,
#         start_time: int,
#         end_time: int,
#         normalize,
#         scaler,
#         seq_length,
#         target_col="cpu_percent",
#         target_features=None
#     ):
#     """
#     Load recent Prometheus data and prepare sequences for hybrid LSTM/CNN/GRU model prediction.

#     Returns:
#     - X: shape (num_samples, seq_length, num_features)
#     - deployment_names: deployment name for each sequence
#     - timestamps: timestamp of the last step in each sequence
#     """

#     if normalize is None or scaler is None:
#         logs("Normalizer or scaler not provided.")
#         return None, None, None

#     # Load data from Prometheus
#     data = load_microservice_data(
#         deployments=deployments,
#         PROMETHEUS_URL=PROMETHEUS_URL,
#         start_time=start_time,
#         end_time=end_time,
#         step=metadata["data_param"]["step_seconds"]
#     )

#     if data.empty:
#         logs("No data available for prediction.")
#         return None, None, None

#     sequences, deployment_names, timestamps = [], [], []

#     for deploy in deployments:
#         if deploy not in data["deployment"].unique():
#             logs(f"No data for {deploy}, skipping.")
#             continue

#         # Filter deployment data and sort by time
#         deploy_data = (
#             data[data["deployment"] == deploy]
#             .sort_values("timestamp")
#             .reset_index(drop=True)
#         )

#         # Select target and optional additional features
#         if target_features is not None and len(target_features) > 0:
#             feature_columns = target_features
#         else:
#             feature_columns = [target_col]

#         raw_data = deploy_data[feature_columns].values

#         # Apply transforms in the correct order: scaler → normalize
#         try:
#             scaled = scaler.transform(raw_data)
#             normalized = normalize.transform(scaled)
#         except Exception as e:
#             logs(f"Transform failed for {deploy}: {e}")
#             continue

#         # Pad if not enough samples
#         if len(normalized) < seq_length:
#             pad_len = seq_length - len(normalized)
#             padded = np.vstack([np.zeros((pad_len, normalized.shape[1])), normalized])
#             sequences.append(padded)
#             deployment_names.append(deploy)
#             timestamps.append(deploy_data["timestamp"].iloc[-1])
#             logs(f"Padded {deploy} with {pad_len} zero rows.")
#         else:
#             # Sliding window sequences
#             for i in range(len(normalized) - seq_length + 1):
#                 seq = normalized[i:i + seq_length]
#                 sequences.append(seq)
#                 deployment_names.append(deploy)
#                 timestamps.append(deploy_data["timestamp"].iloc[i + seq_length - 1])

#     X = np.array(sequences)
#     return X, deployment_names, timestamps
