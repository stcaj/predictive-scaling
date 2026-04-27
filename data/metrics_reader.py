from datetime import datetime
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_dataset import fix_percents
from utils import drop_useless, fix_dates, logger, plot_error_percentage_comparison, plot_frontend_replicas_comparison, plot_frontend_replicas_difference, plot_frontend_replicas_zoom, plot_frontend_replicas_zoom_min, plot_week_heatmap

# names of the different methods
dir_path = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(dir_path, 'data_preprocessing', 'metric_reader_output')


# Week data locust files

def load_prometheus_data(file_pattern):
    all_files = pd.DataFrame()
    for fname in glob.glob(file_pattern):
        df = pd.read_csv(fname)
        all_files = pd.concat([all_files, df], ignore_index=True)    

    return all_files

def fix_prometheus_stats(prometheus_stats):
    data = prometheus_stats.copy()
    data = data[data["Name"] == "Aggregated"].copy()
    return data

# def fix_prometheus_data(prometheus_data):
#     data = prometheus_data.copy()
#     data = drop_useless(data)
#     return data
def fix_prometheus_data(prometheus_data, locust_data=None, prom_ts_col="timestamp", locust_ts_col="Timestamp", tolerance_before_sec=0):
    data = prometheus_data.copy()
    data = drop_useless(data)

    if locust_data is not None:
        if prom_ts_col not in data.columns:
            raise KeyError(f"Prometheus dataset must contain '{prom_ts_col}'. Columns: {list(data.columns)}")
        if locust_ts_col not in locust_data.columns:
            raise KeyError(f"Locust dataset must contain '{locust_ts_col}'. Columns: {list(locust_data.columns)}")

        data[prom_ts_col] = pd.to_numeric(data[prom_ts_col], errors="coerce")
        locust_ts = pd.to_numeric(locust_data[locust_ts_col], errors="coerce")

        data = data.dropna(subset=[prom_ts_col]).copy()
        locust_ts = locust_ts.dropna()

        if data.empty:
            raise ValueError("Prometheus dataset is empty after timestamp cleanup.")
        if locust_ts.empty:
            raise ValueError("Locust dataset is empty after timestamp cleanup.")

        locust_start = float(locust_ts.min())
        prom_before = len(data)

        data = data[data[prom_ts_col] >= (locust_start - tolerance_before_sec)].copy()
        data = data.sort_values(prom_ts_col).reset_index(drop=True)

        if data.empty:
            raise ValueError("No Prometheus rows left after aligning start to Locust.")

        logger(
            f"Prometheus aligned to Locust start: "
            f"locust_start={datetime.fromtimestamp(locust_start)}, "
            f"{prom_before} -> {len(data)} rows, "
            f"prom_start={datetime.fromtimestamp(data[prom_ts_col].iloc[0])}"
        )

    return data


def fix_locust_data(locust_data):
    data = locust_data.copy()
    data = data[data["Name"] == "Aggregated"].copy()
    data = data.sort_values("Timestamp").reset_index(drop=True)
    data_copy = data.copy()

    # Convert key columns to numeric first
    for col in ["Timestamp", "Requests/s", "95%", "Total Request Count", "Total Failure Count"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop warm-up rows (NaN p95)
    while len(data) > 0 and pd.isna(data["95%"].iloc[0]):
        data = data.iloc[1:].reset_index(drop=True)

    if len(data) < len(data_copy):
        logger(f"Locust warm-up removal: {len(data_copy) - len(data)} rows dropped.")
        logger(f"Locust data fixed: {len(data)} rows remain after warm-up removal.")
        data_copy = data.copy()

    if data.empty:
        raise ValueError("No rows remaining after warm-up removal.")

    # Warn if NaNs still exist in critical columns
    nan_counts = data[["Timestamp", "Requests/s", "95%"]].isna().sum()
    if nan_counts.any():
        logger(
            f"NaNs remain in critical columns after warm-up drop: "
            f"{nan_counts[nan_counts > 0].to_dict()}",
            level="WARNING"
        )

    data = data[(~data["95%"].isna())].copy()
    data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # dt in minutes
    data["dt_min"] = data["Timestamp"].diff() / 60.0
    data.loc[0, "dt_min"] = 0.0
    data["dt_min"] = data["dt_min"].clip(lower=0)

    # baseline normalize counters to start from 0 in this sliced dataset
    counter_cols = ["Total Request Count", "Total Failure Count"]
    for col in counter_cols:
        if col in data.columns and not data[col].isna().all():
            first_val = data[col].iloc[0]
            logger(f"{col}: baseline offset = {first_val}")
            data[col] = data[col] - first_val

    # safe delta columns with reset handling
    for col in counter_cols:
        if col in data.columns:
            diff = data[col].diff()
            diff = diff.where(diff >= 0, data[col])   # handle resets
            data[f"{col}_delta"] = diff.fillna(0)

    # aliases
    data["req_count"] = data["Total Request Count_delta"]
    data["fail_count"] = data["Total Failure Count_delta"]

    # optional cleanup log
    zero_traffic_rows = (data["Requests/s"] <= 0).sum()
    nan_p95_rows = data["95%"].isna().sum()
    if zero_traffic_rows > 0 or nan_p95_rows > 0:
        print(
            f"Zero-traffic rows: {zero_traffic_rows}"
            f"NaN p95 rows: {nan_p95_rows}"
        )
        print(data["Requests/s"].describe())

    logger(f"Locust data fixed: {len(data)} rows remain.")
    return data



def get_p95(fixed_locust_data, fixed_stats=None):
    data = fixed_locust_data.copy()

    data["95%"] = pd.to_numeric(data["95%"], errors="coerce")
    data["req_count"] = pd.to_numeric(data["req_count"], errors="coerce")
    data = data.dropna(subset=["95%", "req_count"])
    data = data[data["req_count"] > 0]

    print("p95 latency:")
    if fixed_stats is not None:
        aggregated_p95 = fixed_stats["95%"].iloc[0]
        print(f"  - Global (Locust aggregated): {aggregated_p95:.0f} ms")

    if data["req_count"].sum() <= 0:
        print("  - Load-weighted avg of per-interval p95: N/A ms")
        return None

    weighted_p95 = (data["95%"] * data["req_count"]).sum() / data["req_count"].sum()
    print(f"  - Load-weighted avg of per-interval p95: {weighted_p95:.2f} ms")

    if fixed_stats is None:
        return float(weighted_p95)
    return float(aggregated_p95)

def get_p99(fixed_locust_data, fixed_stats=None):
    data = fixed_locust_data.copy()

    data["99%"] = pd.to_numeric(data["99%"], errors="coerce")
    data["req_count"] = pd.to_numeric(data["req_count"], errors="coerce")

    data = data.dropna(subset=["99%", "req_count"])
    data = data[data["req_count"] > 0]


    print("p99 latency:")
    if fixed_stats is not None:
        aggregated_p99 = fixed_stats["99%"].iloc[0]
        print(f"  - Global (Locust aggregated): {aggregated_p99:.0f} ms")

    if data["req_count"].sum() <= 0:
        print("  - Load-weighted avg of per-interval p99: N/A ms")
        return None

    weighted_p99 = (data["99%"] * data["req_count"]).sum() / data["req_count"].sum()
    print(f"  - Load-weighted avg of per-interval p99: {weighted_p99:.2f} ms")

    if fixed_stats is None:
        return float(weighted_p99)
    return float(aggregated_p99)


def get_SLA_violations(week_locust_data):
    data = week_locust_data.copy()

    MAX_RESPONSE_TIME_SLA = 1000  # ms
    violations = data[data["95%"] > MAX_RESPONSE_TIME_SLA]

    violation_minutes = float(violations["dt_min"].sum())
    total_minutes = float(data["dt_min"].sum())

    ratio_time = violation_minutes / total_minutes if total_minutes > 0 else np.nan
    ratio_rows = (data["95%"] > MAX_RESPONSE_TIME_SLA).mean()

    print(
        f"SLA violation:"
        f"\n  - Time: {violation_minutes:.2f} min"
        f"\n  - Violation ratio (time): {ratio_time:.4%}"
        f"\n  - Violation ratio (rows): {ratio_rows:.4%}"
    )

    return violation_minutes, total_minutes, ratio_time, ratio_rows

def find_stable_reach(future, target, is_up, stable_points=2):
    vals = future[["timestamp", "available_replicas"]].reset_index(drop=True)

    for i in range(len(vals) - stable_points + 1):
        window = vals.iloc[i:i + stable_points]

        if is_up:
            ok = (window["available_replicas"] >= target).all()
        else:
            ok = (window["available_replicas"] == target).all()

        if ok:
            return int(window.iloc[0]["timestamp"])

    return None


def get_global_stats(tts_by_dep, events, interrupted_by_dep, never_reached_by_dep):
    all_tts = np.array([x for xs in tts_by_dep.values() for x in xs], dtype=float) / 60.0

    total_interrupted = int(sum(interrupted_by_dep.values()))
    total_never_reached = int(sum(never_reached_by_dep.values()))

    global_stats = {
        "Total events": int(len(events)),
        "Total reached": int(len(all_tts)),
        "Total interrupted": total_interrupted,
        "Total never reached": total_never_reached,
        "Total not completed": total_interrupted + total_never_reached,
        "Minutes (mean)": float(all_tts.mean()) if len(all_tts) else np.nan,
        "Minutes (median)": float(np.median(all_tts)) if len(all_tts) else np.nan,
        "Minutes (min)": float(all_tts.min()) if len(all_tts) else np.nan,
        "Minutes (max)": float(all_tts.max()) if len(all_tts) else np.nan,
        "Minutes (p95)": float(np.percentile(all_tts, 95)) if len(all_tts) >= 2 else (float(all_tts.max()) if len(all_tts) == 1 else np.nan),
    }

    print("Time to scale (global, reached only) (min):")
    for k, v in global_stats.items():
        if isinstance(v, float):
            print(f"  - {k}: {'N/A' if np.isnan(v) else f'{v:.2f}'}")
        else:
            print(f"  - {k}: {v}")

    return global_stats

def get_stats_per_deployment(tts_by_dep, events_by_dep, interrupted_by_dep, never_reached_by_dep):
    per_dep_stats = []

    for dep in events_by_dep.keys():
        tts_seconds = tts_by_dep.get(dep, [])
        arr = np.array(tts_seconds, dtype=float) / 60.0 if tts_seconds else np.array([])

        interrupted = int(interrupted_by_dep.get(dep, 0))
        never_reached = int(never_reached_by_dep.get(dep, 0))

        per_dep_stats.append({
            "deployment": dep,
            "count_events": int(events_by_dep.get(dep, 0)),
            "count_reached": int(len(arr)),
            "count_interrupted": interrupted,
            "count_never_reached": never_reached,
            "count_not_completed": interrupted + never_reached,
            "mean_min": float(arr.mean()) if len(arr) else np.nan,
            "median_min": float(np.median(arr)) if len(arr) else np.nan,
            "min_min": float(arr.min()) if len(arr) else np.nan,
            "max_min": float(arr.max()) if len(arr) else np.nan,
            "p95_min": float(np.percentile(arr, 95)) if len(arr) >= 2 else (float(arr.max()) if len(arr) == 1 else np.nan),
        })

    per_dep_df = pd.DataFrame(per_dep_stats).sort_values(
        ["count_not_completed", "mean_min"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)

    print("Time to scale per deployment (min):")
    for _, r in per_dep_df.iterrows():
        min_txt = "N/A" if pd.isna(r["min_min"]) else f"{r['min_min']:.2f}"
        max_txt = "N/A" if pd.isna(r["max_min"]) else f"{r['max_min']:.2f}"
        p95_txt = "N/A" if pd.isna(r["p95_min"]) else f"{r['p95_min']:.2f}"

        print(
            f"  - {r['deployment']}:"
            f"\n    events={int(r['count_events'])}, reached={int(r['count_reached'])}, interrupted={int(r['count_interrupted'])}, never_reached={int(r['count_never_reached'])}"
            f"\n    min={min_txt}, max={max_txt}"
            f"\n    p95={p95_txt}"
        )

    return per_dep_df

def get_time_to_scale(week_data):
    data = week_data.copy()
    data = data.sort_values(["deployment", "timestamp"]).reset_index(drop=True)

    data["desired_diff"] = data.groupby("deployment")["desired_replicas"].diff()
    data["available_diff"] = data.groupby("deployment")["available_replicas"].diff()
    data = data.dropna(subset=["desired_diff", "available_diff"])

    events = data[data["desired_diff"] != 0].copy()
    if events.empty:
        print("Time to scale:\n  - N/A min (no scaling events found)")
        return None

    tts_by_dep = {}
    interrupted_by_dep = {}
    never_reached_by_dep = {}
    events_by_dep = events.groupby("deployment").size().to_dict()
    data_by_dep = {dep: df.reset_index(drop=True) for dep, df in data.groupby("deployment", sort=False)}

    for _, ev in events.iterrows():
        dep = ev["deployment"]
        t0 = int(ev["timestamp"])
        target = float(ev["desired_replicas"])
        is_up = ev["desired_diff"] > 0

        dep_data = data_by_dep.get(dep)
        if dep_data is None or dep_data.empty:
            continue

        future = dep_data[dep_data["timestamp"] > t0].copy()
        if future.empty:
            never_reached_by_dep[dep] = never_reached_by_dep.get(dep, 0) + 1
            continue

        next_desired_change = future[future["desired_diff"] != 0]
        was_interrupted = not next_desired_change.empty

        if was_interrupted:
            next_change_ts = int(next_desired_change.iloc[0]["timestamp"])
            eval_window = future[future["timestamp"] < next_change_ts].copy()
        else:
            eval_window = future.copy()

        if eval_window.empty:
            interrupted_by_dep[dep] = interrupted_by_dep.get(dep, 0) + 1
            continue

        t1 = find_stable_reach(eval_window, target, is_up, stable_points=2)
        if t1 is None:
            if was_interrupted:
                interrupted_by_dep[dep] = interrupted_by_dep.get(dep, 0) + 1
            else:
                never_reached_by_dep[dep] = never_reached_by_dep.get(dep, 0) + 1
            continue

        tts = t1 - t0
        if tts >= 0:
            tts_by_dep.setdefault(dep, []).append(tts)

    if not tts_by_dep:
        print("Time to scale:\n  - N/A min (events found, but none reached target)")
        return None
    
    # get stats per dep and global
    per_dep_df = get_stats_per_deployment(
        tts_by_dep,
        events_by_dep,
        interrupted_by_dep,
        never_reached_by_dep
    )

    global_stats = get_global_stats(
        tts_by_dep,
        events,
        interrupted_by_dep,
        never_reached_by_dep
    )
    return per_dep_df, global_stats




def get_total_requests(fixed_locust_data):
    data = fixed_locust_data.copy().sort_values("Timestamp").reset_index(drop=True)
    data["req_count"] = pd.to_numeric(data["req_count"], errors="coerce")
    data = data.dropna(subset=["req_count"])

    total_req = data["req_count"][data["req_count"] >= 0].sum()
    return int(round(total_req))


def get_errors(fixed_locust_data):
    data = fixed_locust_data.copy().sort_values("Timestamp").reset_index(drop=True)

    data["fail_count"] = pd.to_numeric(data["fail_count"], errors="coerce")
    data["Total Failure Count"] = pd.to_numeric(data["Total Failure Count"], errors="coerce")
    data = data.dropna(subset=["fail_count", "Total Failure Count"])

    total_errors = data["fail_count"][data["fail_count"] >= 0].sum()
    sum_diffs = int(round(total_errors)) if not pd.isna(total_errors) else None
    last_value = int(round(data["Total Failure Count"].iloc[-1]))

    req = get_total_requests(fixed_locust_data)
    err_rate = (sum_diffs / req) if (req and sum_diffs is not None) else np.nan

    print(
        f"Errors:"
        f"\n  - {sum_diffs} (last normalized counter value: {last_value})"
        f"\n  - Error rate: {err_rate:.4%}" if np.isfinite(err_rate) else "\n  - Error rate: N/A"
    )
    return err_rate


def get_replica_minutes(prometheus_data):
    data = prometheus_data.copy()
    data = data.sort_values(["deployment", "timestamp"])    # sort by deployment and timestamp
    data["dt_min"] = data.groupby("deployment")["timestamp"].diff().fillna(0) / 60 # difference int timestamps per deployment

    data["replica_minutes"] = data["available_replicas"] * data["dt_min"]
    # sum replica-minutes per deployment
    replica_minutes_per_dep = (
        data.groupby("deployment")["replica_minutes"]
          .sum()
          .reset_index()
          .rename(columns={"replica_minutes": "replica_minutes_total"})
    )

    # total replica-minutes
    total_replica_minutes = replica_minutes_per_dep["replica_minutes_total"].sum()
    # print results
    print("Replica-minutes:")
    print(f"  - All Replica-minutes: {total_replica_minutes:.2f} min")
    print("  - Per deployment:")
    for _, row in replica_minutes_per_dep.iterrows():
        print(f"    - {row['deployment']}: {row['replica_minutes_total']:.2f} min")

    return total_replica_minutes

def trim_locust_by_prometheus_dates(prometheus_dataset, locust_dataset):
    if "timestamp_real" not in prometheus_dataset.columns:
        raise KeyError("Prometheus dataset must contain 'timestamp_real' column.")
    if "Timestamp" not in locust_dataset.columns:
        raise KeyError("Locust dataset must contain 'Timestamp' column.")

    prom = prometheus_dataset.copy()
    prom["_date"] = pd.to_datetime(prom["timestamp_real"], unit="s").dt.date
    selected_dates = prom["_date"].drop_duplicates()

    loc = locust_dataset.copy()
    loc["_date"] = pd.to_datetime(loc["Timestamp"], unit="s").dt.date

    locust_data = loc[loc["_date"].isin(selected_dates)].copy().reset_index(drop=True)
    locust_data = locust_data.drop(columns=["_date"], errors="ignore")

    if locust_data.empty:
        raise ValueError("No rows left in locust after trimming by Prometheus selected dates.")

    return locust_data

def trim_locust_by_prometheus_intervals(
    prometheus_dataset,
    locust_dataset,
    prom_ts_col="timestamp_real",
    locust_ts_col="Timestamp",
    gap_threshold_sec=300,
    tolerance_sec=30
):
    if prom_ts_col not in prometheus_dataset.columns:
        raise KeyError(f"Prometheus dataset must contain '{prom_ts_col}' column.")
    if locust_ts_col not in locust_dataset.columns:
        raise KeyError(f"Locust dataset must contain '{locust_ts_col}' column.")

    prom = prometheus_dataset.copy().sort_values(prom_ts_col).reset_index(drop=True)
    loc = locust_dataset.copy().sort_values(locust_ts_col).reset_index(drop=True)

    prom[prom_ts_col] = pd.to_numeric(prom[prom_ts_col], errors="coerce")
    loc[locust_ts_col] = pd.to_numeric(loc[locust_ts_col], errors="coerce")

    prom = prom.dropna(subset=[prom_ts_col]).copy()
    loc = loc.dropna(subset=[locust_ts_col]).copy()

    if prom.empty:
        raise ValueError("Prometheus dataset is empty after timestamp cleanup.")
    if loc.empty:
        raise ValueError("Locust dataset is empty after timestamp cleanup.")

    # nájdi súvislé bloky v Prometheovi podľa reálneho času
    prom["dt_diff"] = prom[prom_ts_col].diff().fillna(0)
    prom["block"] = (prom["dt_diff"] > gap_threshold_sec).cumsum()

    intervals = (
        prom.groupby("block")[prom_ts_col]
        .agg(["min", "max"])
        .reset_index(drop=True)
    )

    loc_parts = []

    for _, row in intervals.iterrows():
        start_ts = row["min"] - tolerance_sec
        end_ts = row["max"] + tolerance_sec

        part = loc[
            (loc[locust_ts_col] >= start_ts) &
            (loc[locust_ts_col] <= end_ts)
        ].copy()

        if not part.empty:
            loc_parts.append(part)

    if not loc_parts:
        raise ValueError("No rows left in locust after trimming by Prometheus real-time intervals.")

    locust_data = pd.concat(loc_parts, ignore_index=True)
    locust_data = locust_data.drop_duplicates().sort_values(locust_ts_col).reset_index(drop=True)

    logger(
        f"Locust trimmed by Prometheus intervals: "
        f"{len(intervals)} interval(s), "
        f"{len(locust_dataset)} -> {len(locust_data)} rows"
    )

    return locust_data

def trim_prometheus_by_selected_days(prometheus_dataset, selected_days):
    if "day_of_week" not in prometheus_dataset.columns:
        raise KeyError(
            f"Prometheus dataset must contain 'day_of_week'. Columns: {list(prometheus_dataset.columns)}"
        )

    if "timestamp" not in prometheus_dataset.columns:
        raise KeyError(
            f"Prometheus dataset must contain 'timestamp'. Columns: {list(prometheus_dataset.columns)}"
        )

    prom = prometheus_dataset.copy().sort_values("timestamp").reset_index(drop=True)
    prom["day_of_week"] = pd.to_numeric(prom["day_of_week"], errors="coerce")
    prom_data = prom[prom["day_of_week"].isin(selected_days)].copy().reset_index(drop=True)

    if prom_data.empty:
        raise ValueError(
            f"No rows left in prometheus after filtering by day_of_week={selected_days}"
        )

    logger(
        f"Prometheus trimmed by day_of_week={selected_days}: "
        f"{len(prometheus_dataset)} -> {len(prom_data)} rows, "
        f"range={datetime.fromtimestamp(prom_data['timestamp_real'].iloc[0])} - "
        f"{datetime.fromtimestamp(prom_data['timestamp_real'].iloc[-1])}"
    )

    return prom_data

def add_relative_time(dataset, ts_col, new_col="relative_min"):
    data = dataset.copy().sort_values(ts_col).reset_index(drop=True)
    data[new_col] = (data[ts_col] - data[ts_col].iloc[0]) / 60.0
    return data

def fix_gap(dataset, ts_col="timestamp", new_col="relative_min", max_gap_min=1.0):
    data = dataset.copy().sort_values(ts_col).reset_index(drop=True)

    data["dt"] = pd.to_numeric(data[ts_col], errors="coerce").diff() / 60.0
    data.loc[0, "dt"] = 0.0

    # záporné alebo nulové anomálie oprav
    data.loc[data["dt"] < 0, "dt"] = np.nan
    data["dt"] = data["dt"].fillna(max_gap_min)

    # veľké diery zrež
    data["dt_fixed"] = data["dt"].clip(upper=max_gap_min)
    data[new_col] = data["dt_fixed"].cumsum()

    return data.drop(columns=["dt", "dt_fixed"])


def resolve_one_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matched pattern: {pattern} -> {matches}")
    return matches[0]


def plot_plots(locust_datasets, prometheus_datasets, loaded_labels, name="comparison"):
    # plot_error_percentage_comparison(
    #     locust_datasets=locust_datasets,
    #     labels=loaded_labels,
    #     file_name=f"{name}_all_methods_error_percentage",
    #     title="Porovnanie percenta chýb počas testu",
    #     # rolling_window=5
    # )
    # plot_frontend_replicas_comparison(
    #     prometheus_datasets=prometheus_datasets,
    #     labels=loaded_labels,
    #     file_name=f"{name}_frontend_replicas_comparison.png",
    # )
    # plot_frontend_replicas_comparison(
    #     prometheus_datasets=prometheus_datasets,
    #     labels=loaded_labels,
    #     file_name=f"{name}_recommendationservice_available_replicas_comparison.png",
    #     which_replika="available",
    #     microservice="recommendationservice",
    # )

    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_frontend_available_upscale_zoom_min",
        xlim=(360, 540),
        which_replika="available",
    )

    
    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_frontend_desired_upscale_zoom_min",
        xlim=(360, 540),
        which_replika="desired",
    )

    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_frontend_available_downscale_zoom_min",
        xlim=(1320, 1440),
        which_replika="available",
        legend_loc="upper right"
    )   

    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_frontend_desired_downscale_zoom_min",
        xlim=(1320, 1440),
        which_replika="desired",
        legend_loc="upper right"
    )  
    
    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_recommendationservice_available_upscale_zoom_min",
        xlim=(560, 800),
        which_replika="available",
        microservice="recommendationservice",
        legend_loc="lower left"
    )

    plot_frontend_replicas_zoom_min(
        prometheus_datasets,
        loaded_labels,
        file_name=f"{name}_recommendationservice_desired_upscale_zoom_min",
        xlim=(560, 800),
        which_replika="desired",
        microservice="recommendationservice",
        legend_loc="lower left"
    )    

    # plot_frontend_replicas_difference(
    #     prometheus_datasets,
    #     loaded_labels,
    #     baseline_label="HPA",
    #     file_name=f"{name}_frontend_available_difference_vs_hpa_a",
    #     which_replika="available",
    # )

def calculate_global_metrics(files, prometheus_datasets, locust_datasets, name = ""):
    print("===============Plotting frontend for comparison===============")
    all_datasets = []
    loaded_labels = list(files.keys())
    for i, label in enumerate(loaded_labels):
        data = prometheus_datasets[i].copy()
        data = data[data["deployment"] == "frontend"].copy()
        data["deployment"] = label
        all_datasets.append(data)

    plot_plots(locust_datasets, prometheus_datasets, loaded_labels, name=f"{name}_comparison")

    if all_datasets:
        global_frontend = pd.concat(all_datasets, ignore_index=True)

        order = list(files.keys())
        global_frontend["deployment"] = pd.Categorical(
            global_frontend["deployment"],
            categories=order,
            ordered=True
        )

        # plot_week_heatmap(
        #     global_frontend,
        #     file_name="Global_frontend_heatmap",
        #     y_label="Škálovacia metóda"
        # )

        logger("Global frontend plots created.")

def get_logs_in_csv_range(csv_df, log_path):
    df = csv_df.copy()

    if "timestamp_real" not in df.columns:
        raise KeyError("CSV dataframe must contain 'timestamp_real' column.")

    df["log_dt"] = (
        pd.to_datetime(df["timestamp_real"], unit="s", utc=True)
        .dt.tz_convert("Europe/Bratislava")
    )

    start_dt = df["log_dt"].min()
    end_dt = df["log_dt"].max()

    pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    filtered = []

    filtered_lines = 0
    kept_lines = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue

            log_dt = pd.Timestamp(
                datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            ).tz_localize("Europe/Bratislava")

            if start_dt <= log_dt <= end_dt:
                filtered.append(line)
                kept_lines += 1
            else:
                filtered_lines += 1

    logger(f"Filtered lines: {filtered_lines}, Kept lines: {kept_lines}")
    return filtered

# def get_scale_events(predictions):
#     upscale = sum("scaling up" in str(row).lower() for row in predictions)
#     downscale = sum("scaling down" in str(row).lower() for row in predictions)

#     print(f"Scaling events")
#     print(f"  - Upscale events: {upscale}")
#     print(f"  - Downscale events: {downscale}")

#     return upscale, downscale

def get_scale_events(predictions):
    upscale = 0
    downscale = 0
    max_diff = 0
    multi_step_events = 0

    pattern = re.compile(r"(\d+)\s*->\s*(\d+)")

    for row in predictions:
        row_str = str(row).lower()

        if "scaling up" in row_str:
            upscale += 1
        elif "scaling down" in row_str:
            downscale += 1
        else:
            continue

        # extrakcia X -> Y
        match = pattern.search(row_str)
        if match:
            before = int(match.group(1))
            after = int(match.group(2))
            diff = abs(after - before)

            max_diff = max(max_diff, diff)

            if diff > 1:
                multi_step_events += 1


    print(f"Scaling events")
    print(f"  - Upscale events: {upscale}")
    print(f"  - Downscale events: {downscale}")
    print(f"  - Max scaling step: {max_diff}")
    print(f"  - Events with step > 1: {multi_step_events}")

    return upscale, downscale

def get_scale_events_from_prometheus(prometheus_data):
    data = prometheus_data.copy()

    required_cols = ["deployment", "timestamp", "desired_replicas"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise KeyError(f"Prometheus dataset must contain columns {required_cols}. Missing: {missing}")

    data = data.sort_values(["deployment", "timestamp"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["deployment", "timestamp"])

    data["desired_replicas"] = pd.to_numeric(data["desired_replicas"], errors="coerce")
    data = data.dropna(subset=["desired_replicas"])

    # diff
    data["desired_diff"] = data.groupby("deployment")["desired_replicas"].diff()

    events = data.dropna(subset=["desired_diff"]).copy()

    upscale = int((events["desired_diff"] > 0).sum())
    downscale = int((events["desired_diff"] < 0).sum())

    # 🔥 veľkosť scalingu (absolútna hodnota)
    events["abs_diff"] = events["desired_diff"].abs()

    max_diff = events["abs_diff"].max()
    mean_diff = events["abs_diff"].mean()

    # či sa niekedy škálovalo o viac než 1
    multi_step_events = int((events["abs_diff"] > 1).sum())

    print(f"Scaling events")
    print(f"  - Upscale events: {upscale}")
    print(f"  - Downscale events: {downscale}")
    print(f"  - Max scaling step: {max_diff}")
    print(f"  - Events with step > 1: {multi_step_events}")

    return upscale, downscale

def calculate_files_metrics(files, method_name = "All", days = "all"):
    
    prometheus_datasets = []
    locust_datasets = []
    all_attributes = {}

    for name, paths in files.items():
        print(f"\n==============={name}===============")

        loc_pattern = os.path.join(dir_path, paths["locust_week"])
        prom_pattern = os.path.join(dir_path, paths["prometheus"])
        stat_pattern = os.path.join(dir_path, paths["locust_stat"])

        try:
            loc_path = resolve_one_file(loc_pattern)
            stat_path = resolve_one_file(stat_pattern)
        
            loc_raw = pd.read_csv(loc_path)
            prom_raw = load_prometheus_data(prom_pattern)
            stat_raw = pd.read_csv(stat_path)
        except Exception as e:
            print(f"One of the data files doesn't exist or cannot be read: {e}")
            continue

        fixed_locust = fix_locust_data(loc_raw)
        fixed_prometheus = fix_prometheus_data(prom_raw, fixed_locust)

        fixed_prometheus = fix_dates(fixed_prometheus, save_fixed=False)
        fixed_prometheus = trim_prometheus_by_selected_days(fixed_prometheus, paths["selected_days"])

        fixed_locust = trim_locust_by_prometheus_intervals(
            fixed_prometheus,
            fixed_locust,
            gap_threshold_sec=300,
            tolerance_sec=120
        )

        # re-fix after trim so counters start from 0 for the selected interval
        fixed_locust = fix_locust_data(fixed_locust)

        fixed_prometheus = fix_gap(fixed_prometheus, "timestamp")
        fixed_locust = fix_gap(fixed_locust, "Timestamp")

        if paths["fix_percents"]:
            fixed_prometheus = fix_percents(fixed_prometheus)
    
        prometheus_datasets.append(fixed_prometheus)
        locust_datasets.append(fixed_locust)
        if "predictions" in paths:
            pred_path = resolve_one_file(os.path.join(dir_path, paths["predictions"]))
            predictions = get_logs_in_csv_range(fixed_prometheus, pred_path)

        save_fixed_outputs(
            out_dir=os.path.join(OUT_DIR, days),
            key=name,
            locust_df=fixed_locust,
            prometheus_df=fixed_prometheus,
            predictions=predictions if "predictions" in paths else None
        )

        # plot_week_heatmap(fixed_prometheus, file_name=f"{name}_week_heatmap")
        print("----------------------------")
        p95 = get_p95(fixed_locust)
        p99 = get_p99(fixed_locust)
        _, _, sla_ratio_time, _ = get_SLA_violations(fixed_locust)
        errors = get_errors(fixed_locust)
        replica_minutes = get_replica_minutes(fixed_prometheus)

        if "predictions" in paths:
            upscale, downscale = get_scale_events(predictions)
        else:
            upscale, downscale = get_scale_events_from_prometheus(fixed_prometheus)

        all_attributes[name] = {
            "p95": p95,
            "p99": p99,
            "sla_ratio_time": sla_ratio_time,
            "errors": errors,
            "replica_minutes": replica_minutes,
            "upscale": upscale,
            "downscale": downscale,
        }
        # print("----------------------------")
        # get_time_to_scale(fixed_prometheus)
    # make global comparisons
    calculate_global_metrics(
        files=files, prometheus_datasets=prometheus_datasets, locust_datasets=locust_datasets, name=method_name)

    return all_attributes

def write_latex_table(files, all_attributes, name):
    print(f"===============Latex table===============")
    LATEX_END = "\\\\ \\hline"
    logger(f"{name} LATEX table....")
    print("Škálovanie & SLA porušenie (čas) & Chybovosť požiadaviek & Replica-minúty & Počet škálovaní nahor & Počet škálovaní nadol \\\\ \\hline")
    for key in files:
        if key in all_attributes:
            data = all_attributes[key]
            print(
                f"{key:<7}  " 
                f" & {get_percent_text(data['sla_ratio_time'] * 100)} "
                f"& {get_percent_text(data['errors'] * 100)} "
                f"& {add_latex_space(data['replica_minutes'])} "
                f"& {add_latex_space(data['upscale'])} " 
                f"& {add_latex_space(data['downscale'])} "
                f" {LATEX_END}"
            )

def get_percent_text(value, decimals=4):
    text_str = f"{value:.{decimals}f}"
    text_str = text_str.replace(".", "{,}")
    return text_str

def add_latex_space(value):
    text = f"{value:,.0f}"
    text = text.replace(",", r"\,")
    return text

def save_fixed_outputs(out_dir, key, locust_df, prometheus_df, predictions=None):
    safe_key = key.replace(" ", "_").replace("(", "").replace(")", "")
    method_dir = os.path.join(out_dir, safe_key)

    os.makedirs(method_dir, exist_ok=True)

    locust_path = os.path.join(method_dir, "locust_fixed.csv")
    prom_path = os.path.join(method_dir, "prometheus_fixed.csv")

    locust_df.to_csv(locust_path, index=False)
    prometheus_df.to_csv(prom_path, index=False)

    logger(f"Saved Locust to: {locust_path}")
    logger(f"Saved Prometheus to: {prom_path}")

    if predictions is not None:
        pred_path = os.path.join(method_dir, "predictions.log")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.writelines(predictions)

        logger(f"Saved predictions to: {pred_path}")

def main():    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'data_preprocessing', 'plots'), exist_ok=True)
    # ---------------------------------------------------------------------------------------------
    D1_FILES = {
        "Žiadne": {
            "locust_week":      "static_week/locust/locust_stats_history.csv",
            "locust_stat":      "static_week/locust/locust_stats.csv",
            "prometheus":       "static_week/prometheus/*.csv",
            "fix_percents":     True,
            "selected_days":    [3],
        },
        "HPA": {
            "locust_week":      "HPA/locust/locust_stats_history.csv",
            "locust_stat":      "HPA/locust/locust_stats.csv",
            "prometheus":       "HPA/prometheus/*.csv",
            "fix_percents":     True,
            "selected_days":    [3],
        },      
        "Hybrid-A": {
            "locust_week":      "Hybrid/20260331-0402/locust/locust_*_stats_history.csv",
            "locust_stat":      "Hybrid/20260331-0402/locust/locust_*_stats.csv",
            "prometheus":       "Hybrid/20260331-0402/prometheus/*.csv",
            "predictions":      "Hybrid/20260331-0402/hybrid_predictions.log",
            "fix_percents":     False,
            "selected_days":    [0],
        },
        "Hybrid-B": {
            "locust_week":      "Hybrid/20260402-03/locust/locust_*_stats_history.csv",
            "locust_stat":      "Hybrid/20260402-03/locust/locust_*_stats.csv",
            "prometheus":       "Hybrid/20260402-03/prometheus/*.csv",
            "predictions":      "Hybrid/20260402-03/hybrid_predictions.log",
            "fix_percents":     False,
            "selected_days":    [0],
        },      
        # "Hybrid-B": {
        #     "locust_week":      "Hybrid/20260407-10/locust/locust_*_stats_history.csv",
        #     "locust_stat":      "Hybrid/20260407-10/locust/locust_*_stats.csv",
        #     "prometheus":       "Hybrid/20260407-10/prometheus/*.csv",
        #     "predictions":      "Hybrid/20260407-10/hybrid_predictions.log",
        #     "fix_percents":     False,
        #     "selected_days":    [0],
        # },           
        "MLP": {
            "locust_week":      "MLP/20260404-07/locust/locust_*_stats_history.csv",
            "locust_stat":      "MLP/20260404-07/locust/locust_*_stats.csv",
            "prometheus":       "MLP/20260404-07/prometheus/*.csv",
            "predictions":      "MLP/20260404-07/mlp_model_predictions.log",
            "fix_percents":     False,
            "selected_days":    [0],
        },
    }
    print(f"===============Calculating metrics for {list(D1_FILES.keys())}===============")
    ALlD1_files = calculate_files_metrics(files=D1_FILES, method_name="all_d1", days="D1")
    write_latex_table(files=list(D1_FILES.keys()), all_attributes=ALlD1_files, name="All_D1")


    # # ---------------------------------------------------------------------------------------------
    # D3_FILES = {
    #     "Žiadne": {
    #         "locust_week":      "static_week/locust/locust_stats_history.csv",
    #         "locust_stat":      "static_week/locust/locust_stats.csv",
    #         "prometheus":       "static_week/prometheus/*.csv",
    #         "fix_percents":     True,
    #         # "selected_days":    [3],
    #         "selected_days":    [3,4,5],
    #         # "selected_days":    [0,1,2,3,4,5,6],
    #     },
    #     "HPA": {
    #         "locust_week":      "HPA/locust/locust_stats_history.csv",
    #         "locust_stat":      "HPA/locust/locust_stats.csv",
    #         "prometheus":       "HPA/prometheus/*.csv",
    #         "fix_percents":     True,
    #         # "selected_days":    [0,1,2,3,4,5,6],
    #         "selected_days":    [3,4,5],
    #         # "selected_days":    [3],
    #     },
    #     "Hybrid-B": {
    #         "locust_week":      "Hybrid/20260407-10/locust/locust_*_stats_history.csv",
    #         "locust_stat":      "Hybrid/20260407-10/locust/locust_*_stats.csv",
    #         "prometheus":       "Hybrid/20260407-10/prometheus/*.csv",
    #         "predictions":      "Hybrid/20260407-10/hybrid_predictions.log",
    #         "fix_percents":     False,
    #         "selected_days":    [0,1,2],
    #     },        
    #     "MLP": {
    #         "locust_week":      "MLP/20260404-07/locust/locust_*_stats_history.csv",
    #         "locust_stat":      "MLP/20260404-07/locust/locust_*_stats.csv",
    #         "prometheus":       "MLP/20260404-07/prometheus/*.csv",
    #         "predictions":      "MLP/20260404-07/mlp_model_predictions.log",
    #         "fix_percents":     False,
    #         "selected_days":    [0,1,2],
    #     },
    # }
    # print(f"===============Calculating metrics for {list(D3_FILES.keys())}===============")
    # ALlD3_files = calculate_files_metrics(files=D3_FILES, method_name="all_d3", days="D3")
    # write_latex_table(files=list(D3_FILES.keys()), all_attributes=ALlD3_files, name="All_D3")








if __name__ == "__main__":
    main()