import os
import yaml
import requests
import pandas as pd
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Tuple, Optional
from numpy import ndarray
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from my_modules.printable_utils import logs

TZ = ZoneInfo("Europe/Bratislava")
# BASE_TS = int(datetime(2025, 1, 6, 0, 0, 0, tzinfo=TZ).timestamp()) # Day = 0
BASE_TS = int(datetime(2025, 1, 9, 0, 0, 0, tzinfo=TZ).timestamp()) # Day = 3

def count_labels(data: pd.DataFrame, number_of_classes: int) -> None:
    '''
    Count and return number of cases for every class

    Parameters:
    --
    - data: dataframe with loaded data
    - number_of_classes: number of have many classes exists from 0 to n-1
    '''

    for x in range(number_of_classes):
        count_class = (data["class"] == x).sum()
        logs(f"Class {x}: {count_class} samples")

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

def classify(v, class_values):
    if not class_values:
        return None
    for i, threshold in enumerate(class_values):
        if v <= threshold:
            return i
    return len(class_values) - 1

def get_pod_data(
        deployment_name: str, 
        start_time: datetime, 
        end_time: datetime,
        step: int,
        class_values: list,
        PROMETHEUS_API_URL: str, 
        anchor_time: Optional[datetime] = None,
        namespace: str = "default"
    ) -> pd.DataFrame:
    """
    Get data from Prometheus for the given time range for a deployment (all replicas).
    Saves CSV inside dir "pods_metric_data/<deployment_name>".
    """
    # if anchor_time is None this is not used
    anchor_ts = int(anchor_time.timestamp()) if anchor_time is not None else None

    usage_series, memory_dict, total_cpu_limit, total_mem_limit, desired_replicas, available_replicas = load_prometheus_data(
        deployment_name=deployment_name,
        PROMETHEUS_API_URL=PROMETHEUS_API_URL,
        start_ts=int(start_time.timestamp()),
        end_ts=int(end_time.timestamp()),
        step=step,
        namespace=namespace
    )

    rows = []
    for ts_real, cpu_val in usage_series:
        # warmup/filter
        if anchor_ts is not None and ts_real < anchor_ts:
            logs(f"[{deployment_name}] anchor is in the future vs end_time: real:{ts_real}, anchor:{anchor_ts}")
            break

        # choose real or relative date
        if anchor_ts is not None:
            t_rel = ts_real - anchor_ts                 
            sim_ts = BASE_TS + t_rel        
            dt = datetime.fromtimestamp(sim_ts, tz=TZ)  
            ts_out = t_rel
        else:
            dt = datetime.fromtimestamp(ts_real, tz=TZ)
            ts_out = ts_real

        # cpu / mem
        cpu_perc = (cpu_val / total_cpu_limit * 100) if total_cpu_limit > 0 else 0
        mem_val = memory_dict.get(ts_real, None)
        mem_perc = (mem_val / total_mem_limit * 100) if mem_val is not None and total_mem_limit > 0 else 0
        
        # Classification based only on cpu_perc
        # cpu_class = resolve_class(cpu_perc)
        cpu_class = classify(cpu_perc, class_values)
        class_num = cpu_class if cpu_class is not None else (len(class_values) - 1 if class_values else 0)

        # create & append row
        row = {
            "timestamp_real": ts_real,
            "timestamp": ts_out,
            "hour": dt.hour,
            "minute": dt.minute,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "deployment": deployment_name,
            "cpu": cpu_val,
            "cpu_percent": cpu_perc,
            "memory_bytes": mem_val if mem_val is not None else 0,
            "memory_percent": mem_perc,
            "desired_replicas": desired_replicas,
            "available_replicas": available_replicas,
            "class": class_num
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ------------ class from both emory and cpu -------------- saved just in case 
        # mem_class = resolve_class(mem_perc)
        # # get max if classes exists
        # class_candidates = [cls for cls in (cpu_class, mem_class) if cls is not None]
        # class_num = max(class_candidates) if class_candidates else None
        # if not get last one if values exists
        # if class_num is None:
        #     class_num = len(class_values) - 1 if class_values else 0


def load_microservice_data(
        deployments: list, 
        PROMETHEUS_URL: str, 
        start_time: int, 
        end_time: int,
        step: int,
        class_values: list,
        anchor_time: Optional[datetime] = None,
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
            anchor_time=anchor_time,
            class_values=class_values,
            PROMETHEUS_API_URL=PROMETHEUS_URL
        )
        if not df.empty:
            all_dfs.append(df)

    logs("Finished classifying data.")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def preprocess_prometheus(
        deployments: list, 
        data: pd.DataFrame, 
        future_labels: int, 
        features_multiplier: int,
        skipped_steps: int,
        numeric_features: list
    ) -> pd.DataFrame:
    """
    Preprocess Prometheus DataFrame for ML classification with sliding windows.

    Each row = sequence of features from past timesteps + class label from future timestep.

    Parameters
    ---
    - deployments: list of deployment names
    - data: combined DataFrame with all deployments
    - future_labels: number of future steps to predict (for classification we use only 1)
    - features_multiplier: number of past timesteps in input sequence
    - skipped_steps: number of timesteps to skip before predicting

    Returns
    ---
    - preprocessed_data: DataFrame with input features and target class
    """

    # Label encode deployment
    le = LabelEncoder()
    le.fit(deployments)
    data["deployment_id"] = le.fit_transform(data["deployment"])
    features_count = len(numeric_features)

    # loop
    preprocessed_rows = []
    for deploy in deployments:
        # Convert deploy name to its encoded ID
        deploy_id = le.transform([deploy])[0]

        # Filter rows for this deployment
        deploy_data = data[data["deployment_id"] == deploy_id].sort_values("timestamp").reset_index(drop=True)

        if len(deploy_data) < features_multiplier + skipped_steps + future_labels:
            continue  # skip deployments with too little data

        records = deploy_data[numeric_features + ['class']].values.tolist()

        i = 0
        while i < len(records) - features_multiplier - skipped_steps - future_labels + 1:
            # Build input sequence (past features only)
            input_seq = []
            for j in range(i, i + features_multiplier):
                input_seq.extend(records[j][:features_count])

            # Take future class label (classification target)
            labels = [records[j + skipped_steps + k][features_count] for k in range(future_labels)]
            preprocessed_rows.append(input_seq + labels)

            i += 1

    # column names
    columns = []
    for t in range(features_multiplier):
        for f in range(features_count):
            columns.append(f"f{f}_t{t}")
    # label columns
    for k in range(future_labels):
        columns.append(f"label_{k+1}")

    preprocessed_data = pd.DataFrame(preprocessed_rows, columns=columns)

    return preprocessed_data



def load_prometheus(
        metadata, 
        model_name: str, 
        start_time: int, 
        end_time: int,
        anchor_time = None, 
        train_split=0.8,
        smote = False
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Load -> Preprocess -> Split Prometheus data for ML using class labels.

    Parameters
    --
    - metadata: metadata loaded from yaml file
    - model_name: name of the model
    - train_split: fraction of data to use for training

    Returns
    --
    - train_data, train_labels, test_data, test_labels
    """

    logs("Loading data from Prometheus.")
    PROMETHEUS_URL = metadata["prometheus_param"]["url"]
    deployments = metadata["app_deployments"]
    future_labels = metadata["model_param"][f"{model_name}_future_labels"]

    # Load raw data
    data = load_microservice_data(
                deployments=deployments, 
                PROMETHEUS_URL=PROMETHEUS_URL,
                start_time=start_time, 
                end_time=end_time,
                anchor_time=anchor_time,
                step=metadata["data_param"]["step_seconds"],
                class_values=metadata["data_param"]["class_values"]
            )
    
    logs("Finished loading data.")
    
    # logs num of sample for every calss
    count_labels(data=data,number_of_classes=len(metadata["data_param"]["class_values"]))

    # Preprocess for classifier (flatten sequences, keep class column as target)
    preprocessed_data = preprocess_prometheus(
        deployments=deployments,
        data=data,
        future_labels=metadata["model_param"][f"{model_name}_future_labels"],
        features_multiplier=metadata["data_param"]["features_multiplier"],
        skipped_steps=metadata["data_param"]["skipped_steps"],
        numeric_features=metadata["data_param"]["numeric_features"]
    )
    logs("Finished preprocessing data.")

    # Split features and labels
    array_data = preprocessed_data.iloc[:, :-future_labels].to_numpy()
    array_labels = preprocessed_data.iloc[:, -future_labels:].to_numpy()

    if len(array_data) == 0:
        logs("No training samples available after preprocessing. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Train/test split
    try:
        train_data, test_data, train_labels, test_labels = train_test_split(
            array_data,
            array_labels,
            test_size=1-train_split,
            random_state=metadata["model_param"]["random_state"],
            stratify=array_labels
        )
    except ValueError:
        logs("Not enough samples per class to stratify, falling back to random split.")
        train_data, test_data, train_labels, test_labels = train_test_split(
            array_data,
            array_labels,
            test_size=1-train_split,
            random_state=metadata["model_param"]["random_state"]
        )

    logs("Finished train/test split.")
    # return of smote is not true
    if not smote:
        return train_data, train_labels, test_data, test_labels

    # else
    try:
        smote = SMOTE(k_neighbors=2, random_state=42, sampling_strategy={1: 60000, 2: 60000})
        train_data_resampled, train_labels_resampled = smote.fit_resample(train_data, train_labels)
        return train_data_resampled, train_labels_resampled, test_data, test_labels
    except Exception as e:
        logs(f"Could not use SMOTE: ({e}) -> returning data withou it.")
        return train_data, train_labels, test_data, test_labels




def preprocess_prometheus_for_future_prediction(
        deployments: list, 
        data: pd.DataFrame, 
        features_multiplier: int,
        numeric_features: list
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Preprocess Prometheus DataFrame for predicting future steps.

    Each row = sequence of features from past timesteps (no future class label).
    Useful for predicting the next n steps into the future.

    Returns
    ---
    - preprocessed_data: DataFrame with flattened sequences
    - deployment_names: array of deployment names corresponding to each row
    - last_timestamps: array of last timestamps in each sequence
    """

    # Label encode deployment
    le = LabelEncoder()
    le.fit(deployments)
    data["deployment_id"] = le.fit_transform(data["deployment"])
    features_count = len(numeric_features)

    preprocessed_rows = []
    deployment_names = []
    last_timestamps = []

    for deploy in deployments:
        if deploy not in data["deployment"].unique():
            continue
        
        deploy_id = le.transform([deploy])[0]
        deploy_data = data[data["deployment_id"] == deploy_id].sort_values("timestamp").reset_index(drop=True)

        if len(deploy_data) < features_multiplier:
            continue  # skip deployments with too little data

        records = deploy_data[numeric_features].values.tolist()
        timestamps = deploy_data["timestamp"].to_numpy()

        for i in range(len(records) - features_multiplier + 1):
            input_seq = []
            for j in range(i, i + features_multiplier):
                input_seq.extend(records[j])
            preprocessed_rows.append(input_seq)
            deployment_names.append(deploy)
            last_timestamps.append(timestamps[i + features_multiplier - 1])

    # Column names
    columns = []
    for t in range(features_multiplier):
        for f in range(features_count):
            columns.append(f"f{f}_t{t}")

    preprocessed_data = pd.DataFrame(preprocessed_rows, columns=columns)
    return preprocessed_data, np.array(deployment_names), np.array(last_timestamps)




# def load_data_for_prediction(
#         metadata, 
#         start_time, 
#         end_time,
#         anchor_time: Optional[datetime] = None,
#     ) -> Tuple:
#     '''
#     Loads data in format and range for prediction
#     Parameters
#     ---
#     - metada: usufule values
#     - start_time: time from which data are loaded
#     - end_time: time till which data are loaded
#     Returns
#     ---
#     Data, deployment names and timestamps
#     '''
#     deployments = metadata["app_deployments"]
#     # numeric_features = metadata["data_param"]["numeric_features"]
#     numeric_features = metadata["data_param"]["cpu_time_features"]
#     features_multiplier = metadata["data_param"]["features_multiplier"]

#     # Load raw data
#     data = load_microservice_data(
#         deployments=deployments,
#         PROMETHEUS_URL=metadata["prometheus_param"]["url"],
#         start_time=start_time,
#         end_time=end_time,
#         anchor_time=anchor_time,
#         step=metadata["data_param"]["step_seconds"],
#         class_values=metadata["data_param"]["class_values"]
#     )

#     if data.empty:
#         return np.array([]), [], np.array([])

#     # Preprocess for future prediction
#     feature_array, deployment_names, timestamps = preprocess_prometheus_for_future_prediction(
#         deployments=deployments,
#         data=data,
#         features_multiplier=features_multiplier,
#         numeric_features=numeric_features
#     )

#     return feature_array.to_numpy(), deployment_names, timestamps



def add_cyclical_feature(df: pd.DataFrame, col: str, period: int, drop_original: bool = True) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        raise ValueError(f"Column '{col}' not found in dataframe")
    v = out[col].astype(float)
    out[f"{col}_sin"] = np.sin(2 * np.pi * v / period)
    out[f"{col}_cos"] = np.cos(2 * np.pi * v / period)
    if drop_original:
        out = out.drop(columns=[col])
    return out

def add_time_features_from_timestamp(df: pd.DataFrame, numeric_features: list, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Vytvorí iba tie časové featury, ktoré reálne potrebuješ podľa numeric_features,
    a spraví aj sin/cos enkódovanie.
    """
    out = df.copy()
    dt = pd.to_datetime(out[ts_col], unit="s", utc=True).dt.tz_convert("Europe/Bratislava")

    # raw stĺpce len ak treba
    need_hour = ("hour_sin" in numeric_features) or ("hour_cos" in numeric_features)
    need_dow  = ("day_of_week_sin" in numeric_features) or ("day_of_week_cos" in numeric_features)
    need_min  = ("minute_sin" in numeric_features) or ("minute_cos" in numeric_features)
    need_mod  = ("minute_of_day_sin" in numeric_features) or ("minute_of_day_cos" in numeric_features)

    if need_hour and "hour" not in out.columns:
        out["hour"] = dt.dt.hour
    if need_dow and "day_of_week" not in out.columns:
        out["day_of_week"] = dt.dt.dayofweek  # 0..6
    if need_min and "minute" not in out.columns:
        out["minute"] = dt.dt.minute          # 0..59
    if need_mod and "minute_of_day" not in out.columns:
        out["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute  # 0..1439

    # cyclical
    if need_hour:
        out = add_cyclical_feature(out, "hour", 24, drop_original=True)
    if need_dow:
        out = add_cyclical_feature(out, "day_of_week", 7, drop_original=True)
    if need_min:
        out = add_cyclical_feature(out, "minute", 60, drop_original=True)
    if need_mod:
        out = add_cyclical_feature(out, "minute_of_day", 1440, drop_original=True)

    return out

def preprocess_for_prediction(
    deployments: list,
    data: pd.DataFrame,
    features_multiplier: int,
    numeric_features: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vráti:
      X: (N, features_multiplier*len(per_timestep_features) + n_deps)
      deployment_names: (N,)
      last_timestamps: (N,)
    kde one-hot je PRIPOJENÝ RAZ na konci (rovnako ako offline).
    """

    df = data.copy()

    # ---- per timestep features (bez deployment/deployment_id) ----
    per_timestep_features = [f for f in numeric_features if f not in ("deployment_id", "deployment")]
    features_count = len(per_timestep_features)

    # ---- fixed one-hot order ----
    dep_to_idx = {dep: i for i, dep in enumerate(deployments)}
    n_deps = len(deployments)

    X_rows = []
    dep_names = []
    last_ts = []

    for deploy in deployments:
        deploy_data = df[df["deployment"] == deploy].sort_values("timestamp").reset_index(drop=True)
        if len(deploy_data) < features_multiplier:
            continue

        # kontrola stĺpcov
        missing = [c for c in per_timestep_features if c not in deploy_data.columns]
        if missing:
            raise ValueError(f"Missing columns for {deploy}: {missing}")

        records = deploy_data[per_timestep_features].to_numpy()
        timestamps = deploy_data["timestamp"].to_numpy()

        one_hot = np.zeros(n_deps, dtype=float)
        one_hot[dep_to_idx[deploy]] = 1.0

        # Zober iba POSLEDNÉ okno pre každý deployment (na online scaling stačí 1 sample/deployment)
        i = len(records) - features_multiplier
        window = records[i:i + features_multiplier, :features_count].reshape(-1)

        X_rows.append(np.concatenate([window, one_hot]))
        dep_names.append(deploy)
        last_ts.append(timestamps[i + features_multiplier - 1])

    if not X_rows:
        return np.array([]), np.array([]), np.array([])

    return np.vstack(X_rows), np.array(dep_names), np.array(last_ts)


# def load_data_for_prediction(
#     metadata,
#     start_time,
#     end_time,
#     anchor_time: Optional[datetime] = None,
#     input_type: str = "cpu_time_features",  # alebo "cpu_features"
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Online loader pre predikciu:
#       - time feats (podľa configu) -> sin/cos
#       - sliding window flatten
#       - one-hot deployment na konci (raz)
#     """

#     deployments = metadata["app_deployments"]
#     numeric_features = metadata["data_param"][input_type]
#     features_multiplier = metadata["data_param"]["features_multiplier"]

#     data = load_microservice_data(
#         deployments=deployments,
#         PROMETHEUS_URL=metadata["prometheus_param"]["url"],
#         start_time=start_time,
#         end_time=end_time,
#         anchor_time=anchor_time,
#         step=metadata["data_param"]["step_seconds"],
#         class_values=metadata["data_param"]["class_values"],
#     )

#     if data.empty:
#         return np.array([]), np.array([]), np.array([])

#     # urob time sin/cos len ak ich numeric_features vyžadujú
#     data = add_time_features_from_timestamp(data, numeric_features, ts_col="timestamp")
#     logs(f"Columns after time feature generation ({len(data.columns)}):\n{list(data.columns)}")
#     logs(f"Data shape: {data.shape}")
#     logs(f"Columns: {list(data.columns)}")
#     logs(f"First row:\n{data.head(1)}")

#     # sprav okno + one-hot ako offline
#     X, dep_names, last_ts = preprocess_for_prediction(
#         deployments=deployments,
#         data=data,
#         features_multiplier=features_multiplier,
#         numeric_features=numeric_features,
#     )

#     return X, dep_names, last_ts

def debug_data_laoding(X, dep_names, last_ts, deployments, numeric_features, features_multiplier):
    # --- AFTER window + one-hot ---
    logs(f"[PRED] X shape: {X.shape}")
    logs(f"[PRED] dep_names ({len(dep_names)}): \n{dep_names}")
    logs(f"[PRED] last_ts ({len(last_ts)}): {last_ts}")

    if X.size > 0:
        n_deps = len(deployments)
        n_numeric = X.shape[1] - n_deps

        logs(f"[PRED] numeric dim: {n_numeric}, one-hot dim: {n_deps}")

        # feature order (kriticky dôležité)
        per_timestep_features = [f for f in numeric_features if f not in ("deployment_id", "deployment")]
        feature_names = []
        for t in range(features_multiplier):
            for f in per_timestep_features:
                feature_names.append(f"{f}_t{t}")
        for dep in deployments:
            feature_names.append(f"dep_{dep}")

        logs(f"[PRED] Feature order ({len(feature_names)}): \n{feature_names}")

        # ukáž 1. sample rozdelený na numeric vs one-hot
        # logs(f"[PRED] First sample numeric: {X[0][:n_numeric]}")
        logs(f"[PRED] First sample one-hot: {X[0][n_numeric:]}")

        # (voliteľné) mapnutie na názvy - veľké logy, tak len keď chceš
        logs(f"[PRED] First sample mapped: {dict(zip(feature_names, X[0]))}")


def load_data_for_prediction(
    metadata,
    start_time,
    end_time,
    anchor_time: Optional[datetime] = None,
    input_type: str = "cpu_min_time_features",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    deployments = metadata["app_deployments"]
    numeric_features = metadata["data_param"][input_type]
    features_multiplier = metadata["data_param"]["features_multiplier"]

    data = load_microservice_data(
        deployments=deployments,
        PROMETHEUS_URL=metadata["prometheus_param"]["url"],
        start_time=start_time,
        end_time=end_time,
        anchor_time=anchor_time,
        step=metadata["data_param"]["step_seconds"],
        class_values=metadata["data_param"]["class_values"],
    )

    if data.empty:
        logs("[PRED] No data from Prometheus.")
        return np.array([]), np.array([]), np.array([])

    # --- BEFORE time features ---
    # logs(f"[PRED] Raw data shape: {data.shape}")
    # logs(f"[PRED] Raw columns ({len(data.columns)}): \n{list(data.columns)}")

    # 1) time -> sin/cos (len podľa numeric_features)
    data = add_time_features_from_timestamp(data, numeric_features, ts_col="timestamp")

    # --- AFTER time features ---
    # logs(f"[PRED] After time features shape: {data.shape}")
    # logs(f"[PRED] After time features columns ({len(data.columns)}): \n{list(data.columns)}")

    # okno + one-hot encoding
    X, dep_names, last_ts = preprocess_for_prediction(
        deployments=deployments,
        data=data,
        features_multiplier=features_multiplier,
        numeric_features=numeric_features,
    )

    # debug_data_laoding(X, dep_names, last_ts, deployments, numeric_features, features_multiplier)

    return X, dep_names, last_ts





def load_current_classes(metadata, start_time: datetime, end_time: datetime) -> dict:
    deployments = metadata["app_deployments"]

    df = load_microservice_data(
        deployments=deployments,
        PROMETHEUS_URL=metadata["prometheus_param"]["url"],
        start_time=start_time,
        end_time=end_time,
        step=metadata["data_param"]["step_seconds"],
        class_values=metadata["data_param"]["class_values"],
    )
    if df.empty:
        return {}

    latest = (
        df.sort_values("timestamp_real")
          .groupby("deployment", as_index=False)
          .tail(1)
    )

    return {row["deployment"]: {"class": int(row["class"])} for _, row in latest.iterrows()}













# if __name__ == "__main__":

#     try:
#         BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         with open(os.path.join(BASE, 'config.yaml')) as file:
#             config = yaml.safe_load(file)

#         metadata = config["metadata"]
#     except Exception as e:
#         logs(f"Could not get relevant data: ({e}) -> terminating program.")
#         exit(-1)
    
#     end_time = datetime.now().replace(second=0, microsecond=0)
#     start_time = end_time - timedelta(minutes=9)

#     deployments = metadata["app_deployments"]
#     numeric_features = metadata["data_param"]["numeric_features"]
#     features_multiplier = metadata["data_param"]["features_multiplier"]

#     # Load raw data
#     data = load_microservice_data(
#         deployments=deployments,
#         PROMETHEUS_URL=metadata["prometheus_param"]["url"],
#         start_time=start_time,
#         end_time=end_time,
#         step=metadata["data_param"]["step_seconds"],
#         class_values=metadata["data_param"]["class_values"]
#     )
    
#     # data, deployment_names, _ = load_data_for_prediction(metadata, start_time, end_time)

#     print(data)
