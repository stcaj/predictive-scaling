import glob
import os
import pandas as pd
import numpy as np
from utils import logger, plot_week_heatmap, plot_dataset_overview, split_dataset_by_day, fix_dates, drop_useless, save_dataset

# colors for terminal output
INFO_COLOR = '\033[92m'
ERROR_COLOR = '\033[91m'
WHITE_COLOR = '\033[0m'

# directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_preprocessing')
STATIC_PROMETHEUS_DIR = os.path.join(BASE_DIR, 'static_week', 'prometheus')
SYNTHETIC_YEAR_DIR = os.path.join(BASE_DIR, 'synthetic_year')
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
WARMUP_FIX_DIR = os.path.join(DATA_DIR, 'warmup_fixed')
MISSING_DATA_FIX_DIR = os.path.join(DATA_DIR, 'missing_data_fixed')
STATIC_PLOT_DIR = os.path.join(PLOT_DIR, 'static_week')

STATIC_LOCUST_FILE = os.path.join(BASE_DIR, 'static_week', 'locust', 'locust_stats_history.csv')

BASE_YEAR = 2024
START_TIMESTAMP = 1704063600
END_TIMESTAMP = 1704668400
WARMUP_PERIOD_TIMESTAMP = 1763986759 

COLS_TO_AVG = [
    "cpu",
    "cpu_percent",
    "memory_bytes",
    "memory_percent",
]

# -----------------------------------------
# Data loading functions
def load_prometheus():
    df = os.path.join(STATIC_PROMETHEUS_DIR, '*.csv')
    dataset = pd.DataFrame()
    logger(f"Loading dataset from: {df}")

    for fname in glob.glob(df):
        with open(fname, 'r') as infh:
            df = pd.read_csv(infh)
            dataset = pd.concat([dataset, df], ignore_index=True)
            logger(f"Loaded data from file: {os.path.split(fname)[-1]}")

    fixed_dataset = fix_prometheus_data(dataset)
    return fixed_dataset


def load_locust():
    try:
        dataset = pd.read_csv(STATIC_LOCUST_FILE)
        aggregated_dataset = dataset[dataset["Name"] == "Aggregated"].copy()
        return aggregated_dataset[2:]
    except:
        logger(f"Loading file exception: {STATIC_LOCUST_FILE}", level="error")


# placeholder if I choose to combine prometheus and locust data
def combine_data(prometheus_dataset, locust_dataset):
    pass


# -----------------------------------------
# Data fixing functions
def fix_prometheus_data(prometheus_data):
    data = prometheus_data.copy()
    before = data[['deployment','timestamp']].duplicated().sum()
    if before > 0:
        data = data.drop_duplicates(subset=['deployment', 'timestamp'])
        after = data[['deployment','timestamp']].duplicated().sum()
        if before > 0:
            logger(f"Prometheus data fixed: {before - after} duplicate rows dropped." )
            logger(f"Prometheus data fixed: {int(len(data))} rows remain (11 micorservices * {len(data) // 11} rows).")
    data = data.sort_values(['deployment', 'timestamp']).reset_index(drop=True)
    return data


# -----------------------------------------
# Impute data functions

def _minutes_range(h1, m1, h2, m2, inclusive=True):
    start = h1 * 60 + m1
    end = h2 * 60 + m2
    if inclusive:
        end += 1
    out = []
    for t in range(start, end):
        out.append((t // 60, t % 60))
    return out


def fill_missing_rows(day, others, start_hm=(0, 1), end_hm=(0, 37), cols_to_avg=None):
    cols_present = [c for c in cols_to_avg if c in day.columns and c in others.columns]
    avg_other = (
        others
        .groupby(["deployment", "hour", "minute"], as_index=False)[cols_present]
        .mean()
    )

    dep_median = {}
    for c in cols_present:
        dep_median[c] = others.groupby("deployment")[c].median()

    tz = "Europe/Bratislava"
    day_start_ts = int(
        pd.to_datetime(day["timestamp"].min(), unit="s", utc=True)
        .tz_convert(tz)
        .normalize()
        .tz_convert("UTC")
        .timestamp()
    )
    mins = _minutes_range(start_hm[0], start_hm[1], end_hm[0], end_hm[1], inclusive=True)

    new_rows = []
    for dep, g in day.groupby("deployment"):
        existing = set(zip(g["hour"].astype(int).tolist(), g["minute"].astype(int).tolist()))
        template = g.sort_values("timestamp").iloc[0].to_dict()

        for (h, m) in mins:
            if (h, m) in existing:
                continue  # už existuje

            row = template.copy()
            row["deployment"] = dep
            row["hour"] = h
            row["minute"] = m

            row["timestamp"] = int(day_start_ts + h * 3600 + m * 60)
            if "day_of_week" in day.columns:
                row["day_of_week"] = int(g["day_of_week"].iloc[0])
            if "month" in day.columns:
                row["month"] = int(g["month"].iloc[0])
            if "date" in day.columns:
                row["date"] = g["date"].iloc[0]

            match = avg_other[(avg_other["deployment"] == dep) & (avg_other["hour"] == h) & (avg_other["minute"] == m)]
            if len(match) == 1:
                for c in cols_present:
                    row[c] = float(match.iloc[0][c])
            else:
                # fallback: median per deployment -> alebo template value
                for c in cols_present:
                    med = dep_median[c].get(dep, np.nan)
                    if pd.isna(med):
                        row[c] = template.get(c, np.nan)
                    else:
                        row[c] = float(med)

            new_rows.append(row)

    if not new_rows:
        logger("No missing Monday rows detected in the requested interval.")
        return day

    day_filled = pd.concat([day, pd.DataFrame(new_rows)], ignore_index=True)
    day_filled = day_filled.sort_values(["timestamp", "deployment"]).reset_index(drop=True)

    logger(f"Data fixed: Filled {len(new_rows)} missing rows ({start_hm[0]:02d}:{start_hm[1]:02d} - {end_hm[0]:02d}:{end_hm[1]:02d}) using averages.")
    return day_filled

def impute_missing_data(dataset):
    df = dataset.copy()
    plot_dataset_overview(df, "before_imputation", MISSING_DATA_FIX_DIR)
    # First, fill missing rows in Monday morning
    split_df = split_dataset_by_day(df, save=False)
    friday = split_df[6].copy()
    others = pd.concat(split_df[:5], ignore_index=True)

    split_df[6] = fill_missing_rows(
        friday,
        others,
        start_hm=(23, 21),
        end_hm=(23, 59),
        cols_to_avg=COLS_TO_AVG,
    )

    df_after = pd.concat(split_df, ignore_index=True).sort_values(["timestamp", "deployment"]).reset_index(drop=True)
    plot_dataset_overview(df_after, "after_imputation", MISSING_DATA_FIX_DIR)
    save_dataset(df_after, "synthetic_week_imputed_missing_data.csv")
    return df_after


def impute_warmup(dataset):
    df = dataset.copy()

    # per-deployment warmup lengths (in minutes)
    WARMUP_MIN = {
        "adservice": 120,
        "cartservice": 2,
    }
    TARGET_DEPS = set(WARMUP_MIN.keys())
    df_before = df.copy()
    plot_dataset_overview(df_before, "before_imputation", WARMUP_FIX_DIR)

    split_df = split_dataset_by_day(df, save=False)
    monday = split_df[0].copy()
    others = pd.concat(split_df[1:], ignore_index=True)

    cols_present = [c for c in COLS_TO_AVG if c in monday.columns and c in others.columns]
    if not cols_present:
        logger("None of COLS_TO_AVG columns exist; skipping warmup imputation.", level="error")
        return df

    # ---------- build a per-row warmup mask for Monday ----------
    mon_mask = pd.Series(False, index=monday.index)

    for dep, warmup_min in WARMUP_MIN.items():
        dep_rows = monday["deployment"] == dep
        if not dep_rows.any():
            continue

        # start time for this deployment within Monday
        dep_start_ts = monday.loc[dep_rows, "timestamp"].min()
        dep_end_ts = dep_start_ts + warmup_min * 60

        mon_mask |= dep_rows & (monday["timestamp"] >= dep_start_ts) & (monday["timestamp"] < dep_end_ts)

    # ---------- compute averages from other days for same (deployment, hour, minute) ----------
    # only for target deps; and only rows that could be used as donors (you can keep hour/minute filter if you want)
    oth_mask = others["deployment"].isin(TARGET_DEPS)

    avg_other = (
        others.loc[oth_mask]
        .groupby(["deployment", "hour", "minute"], as_index=False)[cols_present]
        .mean()
    )

    mon_win = monday.loc[mon_mask].copy()
    mon_win = mon_win.merge(avg_other, on=["deployment", "hour", "minute"], how="left", suffixes=("", "_avg"))

    for c in cols_present:
        avg_c = f"{c}_avg"
        if avg_c not in mon_win.columns:
            continue

        # fallback: per-deployment median from other days (only target deps)
        dep_median = (
            others.loc[oth_mask]
            .groupby("deployment")[c]
            .median()
        )

        mon_win[avg_c] = mon_win[avg_c].fillna(mon_win["deployment"].map(dep_median))
        mon_win[avg_c] = mon_win[avg_c].fillna(mon_win[c])
        mon_win[c] = mon_win[avg_c]
        mon_win.drop(columns=[avg_c], inplace=True, errors="ignore")

    # write back only affected rows + columns
    for c in cols_present:
        monday.loc[mon_mask, c] = mon_win[c].values

    split_df[0] = monday

    logger(
        f"Warmup fixed per deployment: "
        + ", ".join([f"{k}={v}min" for k, v in WARMUP_MIN.items()])
    )

    df_after = pd.concat(split_df, ignore_index=True).sort_values(["timestamp", "deployment"]).reset_index(drop=True)
    plot_dataset_overview(df_after, "after_imputation", WARMUP_FIX_DIR)

    save_dataset(df_after, "synthetic_week_imputed.csv")
    return df_after

def fix_percents(dataset):
    df_after = dataset.copy()

    cpu_request_limits = {
        "adservice": 200,
        "cartservice": 200,
        "checkoutservice": 100,
        "currencyservice": 100,
        "emailservice": 100,
        "frontend": 100,
        "paymentservice": 100,
        "productcatalogservice": 100,
        "recommendationservice": 100,
        "shippingservice": 100,
    }

    memory_request_limits = {
        "adservice": 180,
        "cartservice": 64,
        "checkoutservice": 64,
        "currencyservice": 64,
        "emailservice": 64,
        "frontend": 64,
        "paymentservice": 64,
        "productcatalogservice": 64,
        "recommendationservice": 220,
        "shippingservice": 64,
    }

    df_after["cpu_req"] = df_after["deployment"].map(cpu_request_limits)
    df_after["memory_req"] = df_after["deployment"].map(memory_request_limits)

    if df_after["cpu_req"].isna().any():
        missing = df_after[df_after["cpu_req"].isna()]["deployment"].unique()
        raise ValueError(f"Missing cpu_request_limits for deployments: {missing}")
    replicas = df_after["desired_replicas"].replace(0, 1).fillna(1)
    cpu_req_cores = df_after["cpu_req"] / 1000
    
    # CPU %
    # df_after["cpu_percent"] = (
    #     df_after["cpu"] / (df_after["cpu_req"] * replicas)
    # ) * 100
    df_after["cpu_percent"] = (
        df_after["cpu"] / (cpu_req_cores * replicas)
    ) * 100

    # Memory % (MiB -> bytes)
    df_after["memory_percent"] = (
        df_after["memory_bytes"] / ((df_after["memory_req"] * 1024 * 1024) * replicas)
    ) * 100

    save_dataset(df_after, "synthetic_week_final.csv")
    return df_after

def main():
    try:
        # Create necessary directories
        os.makedirs(DATA_DIR,exist_ok=True)
        os.makedirs(SYNTHETIC_YEAR_DIR,exist_ok=True)
        os.makedirs(PLOT_DIR,exist_ok=True)
        os.makedirs(WARMUP_FIX_DIR,exist_ok=True)
        os.makedirs(MISSING_DATA_FIX_DIR,exist_ok=True)
        os.makedirs(STATIC_PLOT_DIR,exist_ok=True)

        # Loading data
        prometheus_dataset = load_prometheus()                      # Load raw prometheus dataset
        # locust_dataset = load_locust()                              # Load raw locust dataset
        dataset = prometheus_dataset                                # Use only prometheus or combined with locust


        # Preprocessing
        fixed_dataset = fix_dates(dataset)                          # Fix dates to synthetic year
        dropped_dataset = drop_useless(fixed_dataset)               # Drop useless data (non-scaled services, duplicates)
        # plot_dataset_overview(fixed_dataset, "", STATIC_PLOT_DIR)  # Plot overview of static week data
        # plot_week_heatmap(dropped_dataset)                             # Plot heatmap of static week data
        # plot_service_daily(dropped_dataset)                           # Plot per-day service plots for static week data

        # Data imputation
        fixed_percent_dataset = fix_percents(dropped_dataset)        # Fix percent values
        imputed_missing_dateset = impute_missing_data(fixed_percent_dataset)      # Impute missing / warmup data
        imputed_dataset = impute_warmup(imputed_missing_dateset)        # Impute missing / warmup data
    except Exception as e:
        logger(f"An error occurred: {e}", level="error")

if __name__ == "__main__":
    main()