import re
import os
import numpy as np
import tqdm
import glob
import pandas as pd
    
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from my_modules.printable_utils import logs


TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

def compute_epoch_durations(
    lines: List[str],
    start_marker: str = "Training started",
    end_marker: str = "Training finished",
) -> Dict[str, Any]:
    """
    """
    start_time: Optional[datetime] = None
    pairs = []

    for line in lines:
        m = TS_RE.match(line)
        if not m:
            continue  # preskoč riadky bez timestampu
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

        if start_time is None:
            if start_marker in line:
                start_time = ts
        else:
            if end_marker in line:
                sec = (ts - start_time).total_seconds()
                # skeptická kontrola: záporný čas = log cez polnoc / rozbitý order
                if sec < 0:
                    raise ValueError(f"Negative duration detected: {start_time} -> {ts}")
                pairs.append({"start": start_time, "end": ts, "sec": sec})
                start_time = None  # reset na ďalší tréning

    durations = [p["sec"] for p in pairs]
    total = float(sum(durations))
    avg = total / len(durations) if durations else 0.0

    return {
        "durations_sec": durations,
        "count": len(durations),
        "total_sec": total,
        "avg_sec": avg,
        "pairs": pairs,
    }


def load_synthetic_dataset():
    '''
    Load synthetic dataset from dir
    Returns
    ---
    - train_dataset = dataset for offline incremental training of models
    - test_dataset = dataset for final comparing of trained models 
    '''
    BASE_DIR = Path(__file__).resolve().parents[2]
    SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'synthetic_year')

    df = os.path.join(SYNTHETIC_DIR, '*.csv')
    # logs(f"Loading dataset from: {df}")

    files = sorted(glob.glob(df))
    dataset = []
    for fname in tqdm.tqdm(files, desc="Loading dataset", leave=False):
        dataset.append(pd.read_csv(fname))

    test_dataset = dataset[-1]
    train_dataset = pd.concat(dataset[:-1], ignore_index=True)
    return train_dataset, test_dataset

def split_dataset_by_interval(dataset, interval):
    number_of_rows_per_day = 24*60 # assuming 1 row per minute
    number_of_days_per_interval = interval
    number_of_rows_per_interval = int(number_of_rows_per_day * number_of_days_per_interval) # calculate the number of rows for the given interval

    df = dataset.copy()
    interval_datasets = []
    deployments = df["deployment"].unique()
    df_per_deployment = [
        df[df["deployment"] == deployment].sort_values("timestamp").reset_index(drop=True)
        for deployment in deployments
    ]

    for i in range(0, len(df_per_deployment[0]), number_of_rows_per_interval):
        parts = [g.iloc[i:i+number_of_rows_per_interval] for g in df_per_deployment]
        data = pd.concat(parts, ignore_index=True).sort_values(["timestamp","deployment"], ignore_index=True)
        interval_datasets.append(data)

    return interval_datasets

def add_cyclical_feature(
    df: pd.DataFrame,
    col: str,
    period: int,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Convert cyclical column to sin/cos representation.
    Parameters
    ----------
    - df : pd.DataFrame
    - col : str
        Column name (e.g. hour, day_of_week)
    - period : int
        Cycle length (e.g. 24 for hour, 7 for day_of_week)
    - drop_original : bool
        Whether to remove the original column
    """

    out = df.copy()

    if col not in out.columns:
        raise ValueError(f"Column '{col}' not found in dataframe")

    v = out[col].astype(float)

    out[f"{col}_sin"] = np.sin(2 * np.pi * v / period)
    out[f"{col}_cos"] = np.cos(2 * np.pi * v / period)

    if drop_original:
        out = out.drop(columns=[col])

    return out

def data_preprocess(
        deployments: list,
        data: pd.DataFrame,
        future_labels: int,
        features_multiplier: int,
        skipped_steps: int,
        numeric_features: list
    ) -> pd.DataFrame:
    """
    Preprocess DataFrame for ML classification with sliding windows.
    - For each deployment, builds samples using sliding windows over numeric_features.
    - Adds ONE-HOT deployment vector once per sample (not per timestep).
    - Appends future class labels (multi-step possible).
    """

    df = data.copy()

    # --- numeric features used per timestep (must NOT include deployment_id / deployment one-hot)
    per_timestep_features = [f for f in numeric_features if f not in ("deployment_id", "deployment")]
    features_count = len(per_timestep_features)

    # --- one-hot mapping (fixed order = deployments list)
    dep_to_idx = {dep: i for i, dep in enumerate(deployments)}
    n_deps = len(deployments)

    preprocessed_rows = []

    for deploy in tqdm.tqdm(deployments, desc="Deployment preprocessing", leave=False):
        # filter per deployment
        deploy_data = df[df["deployment"] == deploy].sort_values("timestamp").reset_index(drop=True)

        if len(deploy_data) < features_multiplier + skipped_steps + future_labels:
            continue

        # records: per-timestep features + class
        records = deploy_data[per_timestep_features + ["class"]].to_numpy()

        # constant one-hot for this deployment (added once per sample)
        one_hot = np.zeros(n_deps, dtype=float)
        one_hot[dep_to_idx[deploy]] = 1.0

        i = 0
        last_i = len(records) - features_multiplier - skipped_steps - future_labels + 1
        while i < last_i:
            # build flattened window features
            window = records[i:i + features_multiplier, :features_count].reshape(-1)

            # labels from the future
            base = i + features_multiplier - 1 + skipped_steps  
            labels = [int(records[base + k, features_count]) for k in range(future_labels)]

            # input = window + onehot, then labels
            preprocessed_rows.append(np.concatenate([window, one_hot, np.array(labels, dtype=float)]))

            i += 1

    # --- column names
    columns = []
    for t in range(features_multiplier):
        for f_idx, f_name in enumerate(per_timestep_features):
            columns.append(f"{f_name}_t{t}")

    # one-hot columns (added once)
    for dep in deployments:
        columns.append(f"dep_{dep}")

    # label columns
    for k in range(future_labels):
        columns.append(f"label_{k+1}")

    preprocessed_data = pd.DataFrame(preprocessed_rows, columns=columns)
    logs(f"Columns: {columns}")
    return preprocessed_data

def time_split_raw_per_deployment(df: pd.DataFrame, train_split=0.8):
    train_parts, test_parts = [], []
    for dep, g in df.groupby("deployment", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        split = int(n * train_split)
        train_parts.append(g.iloc[:split])
        test_parts.append(g.iloc[split:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def offline_data_preprocess(
        dataset, 
        metadata,
        future_labels, 
        numeric_features, 
        train_split=0.8
    ):
    '''
    Preprocess dataset for offline testing using MLP
    
    Parameters
    ---
    - dataset - dataset
    - metadata - metadata
    - model_name - name of the model to differentiate from config yaml file
    - future_labels - number of predicted labels
    - numeric_features - features to keep for training
    - train_split - how to split dataset -> 0.8 = 80/20
    '''
    df = dataset.copy()
    df_ready = add_cyclical_feature(df, col="hour", period=24)
    df_ready = add_cyclical_feature(df_ready, col="minute", period=60)
    df_ready = add_cyclical_feature(df_ready, col="day_of_week", period=7)
    # deployments = df_ready["deployment"].unique()
    deployments = metadata["app_deployments"]

    # split na raw dátach
    df_train, df_test = time_split_raw_per_deployment(df_ready, train_split=train_split)

    # preprocess zvlášť
    pre_train = data_preprocess(
        deployments=deployments,
        data=df_train,
        future_labels=future_labels,
        features_multiplier=metadata["data_param"]["features_multiplier"],
        skipped_steps=metadata["data_param"]["skipped_steps"],
        numeric_features=numeric_features
    )
    pre_test = data_preprocess(
        deployments=deployments,
        data=df_test,
        future_labels=future_labels,
        features_multiplier=metadata["data_param"]["features_multiplier"],
        skipped_steps=metadata["data_param"]["skipped_steps"],
        numeric_features=numeric_features
    )

    train_data = pre_train.iloc[:, :-future_labels].to_numpy()
    train_labels = pre_train.iloc[:, -future_labels:].to_numpy()
    test_data = pre_test.iloc[:, :-future_labels].to_numpy()
    test_labels = pre_test.iloc[:, -future_labels:].to_numpy()

    logs("Finished TIME train/test split.")
    return train_data, train_labels, test_data, test_labels

def offline_data_preprocess_for_testing(
        dataset,
        metadata,
        future_labels,
        numeric_features,
    ):
    """
    Preprocess dataset for offline testing (predict/evaluate) using the whole file as test set.

    Returns
    ---
    - test_data: np.ndarray (N, features)
    - test_labels: np.ndarray (N, future_labels)
    """
    df = dataset.copy()
    df_ready = add_cyclical_feature(df, col="hour", period=24)
    df_ready = add_cyclical_feature(df_ready, col="minute", period=60)
    df_ready = add_cyclical_feature(df_ready, col="day_of_week", period=7)    
    # deployments = df_ready["deployment"].unique()
    deployments = metadata["app_deployments"]


    # preprocess na celom datasete
    pre_test = data_preprocess(
        deployments=deployments,
        data=df_ready,
        future_labels=future_labels,
        features_multiplier=metadata["data_param"]["features_multiplier"],
        skipped_steps=metadata["data_param"]["skipped_steps"],
        numeric_features=numeric_features
    )

    test_data = pre_test.iloc[:, :-future_labels].to_numpy()
    test_labels = pre_test.iloc[:, -future_labels:].to_numpy()


    logs("Finished preprocessing for TESTING.")
    return test_data, test_labels
