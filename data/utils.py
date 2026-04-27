import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from zoneinfo import ZoneInfo


# colors for terminal output
INFO_COLOR = '\033[92m'
ERROR_COLOR = '\033[91m'
WARNING_COLOR = '\033[93m'
WHITE_COLOR = '\033[0m'
# directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_preprocessing')
STATIC_PROMETHEUS_DIR = os.path.join(BASE_DIR, 'static_week', 'prometheus')
SYNTHETIC_YEAR_DIR = os.path.join(BASE_DIR, 'synthetic_year')
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
WARMUP_FIX_DIR = os.path.join(DATA_DIR, 'warmup_fixed')
MISSING_DATA_FIX_DIR = os.path.join(DATA_DIR, 'missing_data_fixed')
# time constants
BASE_YEAR = 2024
START_TIMESTAMP = 1704063600
END_TIMESTAMP = 1704668400
WARMUP_PERIOD_TIMESTAMP = 1763986759

# -----------------------------------------
# Logging function
def logger(message, level="info"):
    # time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time = datetime.now().strftime("%H:%M:%S")
    if level == "info":
        print(f"[{time}][{INFO_COLOR}INFO{WHITE_COLOR}] {message}")
    elif level == "error":
        print(f"[{time}][{ERROR_COLOR}ERROR{WHITE_COLOR}] {message}")
    elif level == "warning":
        print(f"[{time}][{WARNING_COLOR}WARNING{WHITE_COLOR}] {message}")
    else:
        print(f"[{time}][{WHITE_COLOR}LOG{WHITE_COLOR}] {message}")

# ----------------------------------------- 
# Dataset saving functions
def save_dataset(dataset, filename, debug=True):
    fixed_file_path = os.path.join(DATA_DIR, filename)
    dataset.to_csv(fixed_file_path, index=False)
    if debug:
        logger(f"Saved dataset as: {os.path.split(fixed_file_path)[-1]}")

# ------------------------------
def fix_dates(dataset, save_fixed=True):
    df = dataset.copy()

    if "timestamp" not in df.columns:
        raise KeyError(f"No 'timestamp' column found. Columns: {list(df.columns)}")

    # zachovaj pôvodný čas
    if "timestamp_real" not in df.columns:
        df["timestamp_real"] = df["timestamp"]

    # warmup filter na realnom čase
    df = df[df["timestamp"] >= WARMUP_PERIOD_TIMESTAMP].copy()

    if df.empty:
        raise ValueError("Dataset is empty after applying warmup filter.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # explicit timezone
    base_date = datetime(BASE_YEAR, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("Europe/Bratislava"))
    base_unix = int(base_date.timestamp())

    t0 = int(df["timestamp"].iloc[0])

    # syntetický čas
    df["timestamp"] = base_unix + (df["timestamp"].astype("int64") - t0)

    synthetic_dt = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Europe/Bratislava")

    df["hour"] = synthetic_dt.dt.hour
    df["minute"] = synthetic_dt.dt.minute
    df["day_of_week"] = synthetic_dt.dt.weekday
    df["month"] = synthetic_dt.dt.month

    # filter už nad syntetickým časom
    df = df[df["timestamp"] >= START_TIMESTAMP].copy()
    df = df[df["timestamp"] <= END_TIMESTAMP].copy()

    if df.empty:
        raise ValueError("Dataset is empty after applying START_TIMESTAMP / END_TIMESTAMP filter.")

    if save_fixed:
        save_dataset(df, "synthetic_week_from_monday.csv")

    logger("Dates fixed in dataset - starting from 1.1.2024.")
    return df

# def fix_dates(dataset, save_fixed=True):
#     df = dataset.copy()
#     # pridaj timestamp_real potrebny
#     if "timestamp" not in df.columns:
#         raise KeyError(f"No 'timestamp' column found. Columns: {list(df.columns)}")
#     df["timestamp_real"] = df["timestamp"]
    
#     # skip first rows - warmup period
#     df = df[df["timestamp"] >= WARMUP_PERIOD_TIMESTAMP].copy()

#     df = df.sort_values("timestamp").reset_index(drop=True)
#     base_date = datetime(BASE_YEAR, 1, 1, 0, 0, 0)
#     base_unix = int(base_date.timestamp())

#     t0 = int(df["timestamp"].iloc[0])
#     df["timestamp"] = base_unix + (df["timestamp"].astype("int64") - t0)

#     synthetic_dt = (
#         pd.to_datetime(df["timestamp"], unit="s", utc=True)
#         .dt.tz_convert("Europe/Bratislava")
#     )

#     df["hour"] = synthetic_dt.dt.hour
#     df["minute"] = synthetic_dt.dt.minute
#     df["day_of_week"] = synthetic_dt.dt.weekday
#     df["month"] = synthetic_dt.dt.month

#     df = df[df["timestamp"] >= START_TIMESTAMP].copy()
#     df = df[df["timestamp"] <= END_TIMESTAMP].copy()
#     # save fixed dataset

#     save_dataset(df, "synthetic_week_from_monday.csv")
#     # return fixed dataset
#     logger("Dates fixed in dataset - starting from 1.1.2024.")
#     return df

# ----------------------------------------- 
# Drop useless rows from dataset - non-scaled services, duplicates
def drop_useless(dataset, attributes=None):
    df = dataset.copy()
    # -----------------------------------------
    # Drop non-scaled service (redis-cart)
    # -----------------------------------------
    before = len(df)
    df = df[df["deployment"] != "redis-cart"].copy()
    after = len(df)
    removed = before - after
    if removed > 0:
        logger(f"Data fixed: Dropped redis-cart service: {removed} rows removed.")
    # -------------------------------------------------
    # Drop duplicates where (deployment, hour, minute, day_of_week) repeats
    # Keep the latest sample (max timestamp) per minute
    # -------------------------------------------------
    before_len = len(df)
    df_after = df.sort_values(["deployment", "hour", "minute", "day_of_week", "timestamp"]).reset_index(drop=True) # Sort so that the "latest" timestamp is last within each (deployment, hour, minute, day_of_week)
    df_after = df_after.drop_duplicates(subset=["deployment", "hour", "minute", "day_of_week"], keep="last").reset_index(drop=True) # Drop duplicates keeping the last row per (deployment, hour, minute, day_of_week)
    removed_len = before_len - len(df_after)
    if removed_len > 0:
        logger(f"Data fixed: Dropped {removed_len} duplicate rows.")
    
    if attributes is not None:
        df_after = df_after[attributes].copy()

    return df_after


# -----------------------------------------
# Plotting functions
def plot_service(df_src, title, deployment_name, name_suffix, plot_dir):
    df_local = df_src.copy()
    df_local["dt"] = (
        pd.to_datetime(df_local["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Europe/Bratislava")
    )

    ad = df_local[df_local["deployment"] == deployment_name].copy()
    ad["date"] = ad["dt"].dt.date

    plt.figure(figsize=(14, 6))

    DAY_MAP = {
        0: "Pondelok",
        1: "Utorok",
        2: "Streda",
        3: "Štvrtok",
        4: "Piatok",
        5: "Sobota",
        6: "Nedeľa",
    }
    for day, group in ad.groupby("date"):
        weekday = DAY_MAP[pd.to_datetime(day).weekday()]        
        t = (group["dt"] - group["dt"].dt.normalize()).dt.total_seconds() / 3600.0
        plt.plot(
            t,
            group["cpu_percent"],
            linewidth=1.2,
            alpha=0.85,
            label=weekday
        )
    plt.xlabel("Čas dňa (hodiny)", fontsize=14)
    plt.ylabel("CPU využitie (%)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        fontsize=10,
        # loc="upper left",
        # ncol=2,
        # frameon=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"{deployment_name}_{name_suffix}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def plot_autoscaling_cpu(df_src, name_suffix, dir_suffix=None, time_range=3600.0, x_label="Čas dňa (hodiny)"):
    if dir_suffix is not None:
        plot_dir = os.path.join(PLOT_DIR, dir_suffix)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    else:
        plot_dir = PLOT_DIR
    
    df_local = df_src.copy()
    df_local["dt"] = (
        pd.to_datetime(df_local["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Europe/Bratislava")
    )

    plt.figure(figsize=(14, 6))

    start_time = df_local["dt"].min()

    for deployment, group in df_local.groupby("deployment"):
        # t = (group["dt"] - group["dt"].dt.normalize()).dt.total_seconds() / time_range
        t = (group["dt"] - start_time).dt.total_seconds() / time_range
        plt.plot(
            t,
            group["cpu_percent"],
            linewidth=1.2,
            alpha=0.85,
            label=deployment
        )

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("CPU využitie (%)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        fontsize=10,
        # loc="upper left",
        # ncol=2,
        # frameon=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"all_deployments_{name_suffix}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def sk_day_formatter(x):
    SK_DAY = ["Po", "Ut", "St", "Št", "Pi", "So", "Ne"]
    d = mdates.num2date(x)  # datetime (naive v lokálnom tz Matplotlibu)
    return f"{SK_DAY[d.weekday()]} {d.day:02d}.{d.month:02d}"


def plot_dataset_overview(dataset, file_addon, dir_path):
    df = dataset.copy()
    for dep in df["deployment"].unique():
        plot_service(df, f"CPU % for {dep}", dep, file_addon, dir_path)

def plot_service_daily(dataset, dir_suffix=None):
    if dir_suffix is not None:
        plot_dir = os.path.join(PLOT_DIR, dir_suffix)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    else:
        plot_dir = PLOT_DIR

    plot_autoscaling_cpu(dataset, name_suffix="overview", dir_suffix=dir_suffix, time_range=86400.0, x_label="Čas od začiatku experimentu (deň)")


    day_datasets = split_dataset_by_day(dataset, save=False)
    for i, day_df in enumerate(day_datasets):
        if day_df.empty:
            logger(f"Skipping day {i+1}: empty dataset")
            continue

        day_dt = (
            pd.to_datetime(day_df["timestamp"].iloc[0], unit="s", utc=True)
            .tz_convert("Europe/Bratislava")
        )
        suffix = day_dt.strftime("%Y%m%d")
        plot_autoscaling_cpu(day_df, suffix, dir_suffix=dir_suffix)

def plot_week_heatmap(
    df_src,
    resample_rule="5min",
    agg: str = "mean",
    file_name: str = "week_heatmap",
    y_label: str = "Mikroslužba"
):
    df = df_src.copy()

    if "relative_min" not in df.columns:
        raise KeyError(
            f"Column 'relative_min' not found. "
            f"Run add_relative_time(...) before plot_week_heatmap(). "
            f"Columns: {list(df.columns)}"
        )

    if "deployment" not in df.columns or "cpu_percent" not in df.columns:
        raise KeyError(
            f"Required columns missing. Columns: {list(df.columns)}"
        )

    # kontinuálny čas bez medzier
    df["dt_rel"] = pd.to_datetime(df["relative_min"], unit="m")
    df = df.sort_values(["dt_rel", "deployment"]).reset_index(drop=True)

    # pivot: time x service
    pivot = df.pivot_table(
        index="dt_rel",
        columns="deployment",
        values="cpu_percent",
        aggfunc="mean",
        observed=False,
    )

    # resample na kontinuálnej osi
    if agg == "max":
        pivot = pivot.resample(resample_rule).max()
    else:
        pivot = pivot.resample(resample_rule).mean()

    # zachovanie poradia kategórií, ak je deployment categorical
    if isinstance(df["deployment"].dtype, pd.CategoricalDtype):
        pivot = pivot.reindex(df["deployment"].cat.categories, axis=1)

    # ak po resamplingu vzniknú prázdne riadky, zahodiť ich
    pivot = pivot.dropna(how="all")

    if pivot.empty:
        raise ValueError("Pivot is empty after resampling.")

    services = pivot.columns.tolist()
    Z = pivot.T.values

    # X os v hodinách od začiatku
    times_rel_h = (pivot.index - pivot.index[0]).total_seconds() / 3600.0
    x0 = float(times_rel_h[0])
    x1 = float(times_rel_h[-1])

    plt.figure(figsize=(16, 6))
    ax = plt.gca()

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, -0.5, len(services) - 0.5],
        interpolation="nearest",
        vmin=0,
        vmax=200,
    )

    ax.set_yticks(range(len(services)))
    ax.set_yticklabels(services, fontsize=10)

    ax.set_xlabel("Čas od začiatku experimentu (hodiny)", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
    ax.tick_params(axis="x", labelsize=11)

    # deliace čiary po 24 hodinách
    if x1 >= 24:
        for h in range(24, int(x1) + 1, 24):
            ax.axvline(h, linewidth=0.8, alpha=0.6)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("CPU využitie (%)", fontsize=12)

    plt.tight_layout()

    out_png = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_progression_comparison(
    locust_datasets,
    labels,
    file_name="error_progression_comparison",
    title="Porovnanie priebehu chýb počas testu",
    use_error_rate=False
):
    if len(locust_datasets) != len(labels):
        raise ValueError("locust_datasets and labels must have the same length.")

    plt.figure(figsize=(14, 6))

    for data_raw, label in zip(locust_datasets, labels):
        data = data_raw.copy().sort_values("Timestamp").reset_index(drop=True)

        required_cols = ["Timestamp", "Total Failure Count", "Total Request Count"]
        for col in required_cols:
            if col not in data.columns:
                raise KeyError(f"Missing column '{col}' in dataset '{label}'.")

        data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
        data["Total Failure Count"] = pd.to_numeric(data["Total Failure Count"], errors="coerce")
        data["Total Request Count"] = pd.to_numeric(data["Total Request Count"], errors="coerce")

        data = data.dropna(subset=required_cols).copy()
        if data.empty:
            print(f"[WARNING] Dataset '{label}' is empty after preprocessing. Skipping.")
            continue

        start_ts = data["Timestamp"].iloc[0]
        data["elapsed_min"] = (data["Timestamp"] - start_ts) / 60.0

        data["interval_failures"] = data["Total Failure Count"].diff()
        data["interval_requests"] = data["Total Request Count"].diff()

        # counter reset protection
        data.loc[data["interval_failures"] < 0, "interval_failures"] = np.nan
        data.loc[data["interval_requests"] < 0, "interval_requests"] = np.nan

        data["interval_failures"] = data["interval_failures"].fillna(0)
        data["interval_requests"] = data["interval_requests"].fillna(0)

        data["cumulative_failures"] = data["interval_failures"].cumsum()

        data["interval_error_rate_pct"] = np.where(
            data["interval_requests"] > 0,
            (data["interval_failures"] / data["interval_requests"]) * 100.0,
            np.nan
        )

        if use_error_rate:
            plt.plot(
                data["elapsed_min"],
                data["interval_error_rate_pct"],
                linewidth=1.8,
                label=label
            )
        else:
            plt.plot(
                data["elapsed_min"],
                data["cumulative_failures"],
                linewidth=2,
                label=label
            )

    plt.xlabel("Čas od začiatku testu (min)")

    if use_error_rate:
        plt.ylabel("Chybovosť za interval (%)")
    else:
        plt.ylabel("Kumulatívny počet chýb")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_frontend_replicas_comparison(
    prometheus_datasets,
    labels,
    file_name,
    which_replika=None,
    microservice="frontend",
):
    plt.figure(figsize=(14, 6))

    max_x = 0

    for i in range(len(prometheus_datasets)):
        data = prometheus_datasets[i].copy()
        data = data[data["deployment"] == microservice].copy()

        if data.empty:
            logger(f"No {microservice} data for {labels[i]}", level="WARNING")
            continue

        if "relative_min" not in data.columns:
            raise KeyError(
                f"Column 'relative_min' not found for {labels[i]}"
            )

        data = data.sort_values("relative_min").reset_index(drop=True)
        data["relative_h"] = data["relative_min"] / 60.0
        max_x = max(max_x, data["relative_h"].max())

        # available replicas
        if which_replika is None or which_replika == "available":
            addon = ""
            if which_replika is None:
                addon = " - available"

            plt.step(
                data["relative_h"],
                data["available_replicas"],
                where="post",
                label=f"{labels[i]}{addon}"
            )

        # desired replicas
        if which_replika is None or which_replika == "desired":
            if which_replika is None:
                plt.step(
                    data["relative_h"],
                    data["desired_replicas"],
                    where="post",
                    linestyle="--",
                    label=f"{labels[i]} - desired"
                )
            else:
                plt.step(
                    data["relative_h"],
                    data["desired_replicas"],
                    where="post",
                    label=f"{labels[i]}"
                )

    ax = plt.gca()

    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    plt.grid(True, alpha=0.2)

    if max_x >= 24:
        for h in range(24, int(max_x) + 1, 24):
            ax.axvline(h, linewidth=1.0, alpha=0.5, linestyle=":")

    plt.ylim(bottom=0)
    plt.xlabel("Čas od začiatku experimentu (hodiny)")
    plt.ylabel("Počet replík")

    plt.legend(
        # loc="upper left",
        bbox_to_anchor=(1, 1),
        borderaxespad=0
    )

    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_error_percentage_comparison(
    locust_datasets,
    labels,
    file_name="error_percentage_comparison",
    title="Porovnanie percenta chýb počas testu",
    rolling_window=None
):
    if len(locust_datasets) != len(labels):
        raise ValueError("locust_datasets and labels must have the same length.")

    plt.figure(figsize=(14, 6))
    plotted_any = False

    for data_raw, label in zip(locust_datasets, labels):
        data = data_raw.copy().sort_values("Timestamp").reset_index(drop=True)

        required_cols = ["Total Failure Count", "Total Request Count"]
        for col in required_cols:
            if col not in data.columns:
                raise KeyError(f"Missing column '{col}' in dataset '{label}'.")

        data["Total Failure Count"] = pd.to_numeric(data["Total Failure Count"], errors="coerce")
        data["Total Request Count"] = pd.to_numeric(data["Total Request Count"], errors="coerce")

        if "relative_min" in data.columns:
            data["elapsed_h"] = pd.to_numeric(data["relative_min"], errors="coerce") / 60.0
        else:
            data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
            start_ts = data["Timestamp"].iloc[0]
            data["elapsed_h"] = (data["Timestamp"] - start_ts) / 3600.0

        data = data.dropna(subset=["elapsed_h", "Total Failure Count", "Total Request Count"]).copy()
        if data.empty:
            print(f"[WARNING] Dataset '{label}' is empty after preprocessing. Skipping.")
            continue

        data["interval_failures"] = data["Total Failure Count"].diff()
        data["interval_requests"] = data["Total Request Count"].diff()

        data.loc[data["interval_failures"] < 0, "interval_failures"] = np.nan
        data.loc[data["interval_requests"] < 0, "interval_requests"] = np.nan

        data["interval_failures"] = data["interval_failures"].fillna(0)
        data["interval_requests"] = data["interval_requests"].fillna(0)

        data["interval_error_pct"] = np.where(
            data["interval_requests"] > 0,
            (data["interval_failures"] / data["interval_requests"]) * 100.0,
            np.nan
        )

        if rolling_window is not None and rolling_window > 1:
            data["plot_error_pct"] = data["interval_error_pct"].rolling(
                window=rolling_window,
                min_periods=1
            ).mean()
        else:
            data["plot_error_pct"] = data["interval_error_pct"]

        plt.plot(
            data["elapsed_h"],
            data["plot_error_pct"],
            linewidth=1.8,
            label=label
        )
        plotted_any = True

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(12))

    xmax = ax.get_xlim()[1]
    if xmax >= 24:
        for h in range(24, int(xmax) + 1, 24):
            ax.axvline(h, linewidth=0.8, alpha=0.6)

    plt.xlabel("Čas od začiatku testu (hodiny)")
    plt.ylabel("Percento chýb (%)")
    plt.grid(True, alpha=0.3)

    if plotted_any:
        legend = plt.legend(
            # loc="upper left",
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            title="Technika škálovania"
        )
        # legend.get_title().set_weight("bold")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cumulative_error_percentage_comparison(
    locust_datasets,
    labels,
    file_name="cumulative_error_percentage_comparison",
    title="Porovnanie kumulatívneho percenta chýb počas testu",
):
    """
    Vykreslí jeden spoločný graf kumulatívneho percenta chýb v čase pre viacero metód.

    cumulative_error_pct(t) =
        cumulative_failures(t) / cumulative_requests(t) * 100
    """
    if len(locust_datasets) != len(labels):
        raise ValueError("locust_datasets and labels must have the same length.")

    plt.figure(figsize=(14, 6))
    plotted_any = False

    for data_raw, label in zip(locust_datasets, labels):
        data = data_raw.copy().sort_values("Timestamp").reset_index(drop=True)

        required_cols = ["Timestamp", "Total Failure Count", "Total Request Count"]
        for col in required_cols:
            if col not in data.columns:
                raise KeyError(f"Missing column '{col}' in dataset '{label}'.")

        data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
        data["Total Failure Count"] = pd.to_numeric(data["Total Failure Count"], errors="coerce")
        data["Total Request Count"] = pd.to_numeric(data["Total Request Count"], errors="coerce")

        data = data.dropna(subset=required_cols).copy()
        if data.empty:
            print(f"[WARNING] Dataset '{label}' is empty after preprocessing. Skipping.")
            continue

        start_ts = data["Timestamp"].iloc[0]
        data["elapsed_min"] = (data["Timestamp"] - start_ts) / 60.0

        data["interval_failures"] = data["Total Failure Count"].diff()
        data["interval_requests"] = data["Total Request Count"].diff()

        # ochrana proti resetom counterov
        data.loc[data["interval_failures"] < 0, "interval_failures"] = np.nan
        data.loc[data["interval_requests"] < 0, "interval_requests"] = np.nan

        data["interval_failures"] = data["interval_failures"].fillna(0)
        data["interval_requests"] = data["interval_requests"].fillna(0)

        data["cumulative_failures"] = data["interval_failures"].cumsum()
        data["cumulative_requests"] = data["interval_requests"].cumsum()

        data["cumulative_error_pct"] = np.where(
            data["cumulative_requests"] > 0,
            (data["cumulative_failures"] / data["cumulative_requests"]) * 100.0,
            np.nan
        )

        plt.plot(
            data["elapsed_min"],
            data["cumulative_error_pct"],
            linewidth=2,
            label=label
        )
        plotted_any = True

    plt.xlabel("Čas od začiatku testu (min)")
    plt.ylabel("Kumulatívne percento chýb (%)")
    # plt.title(title)
    plt.grid(True, alpha=0.3)

    if plotted_any:
        legend = plt.legend(
            # loc="upper left",
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            title="Technika škálovania"
        )
        # legend.get_title().set_weight("bold")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_microservice_series(prometheus_datasets, labels, replica_col="available_replicas", microservice="frontend"):
    series = {}

    for i, label in enumerate(labels):
        data = prometheus_datasets[i].copy()
        data = data[data["deployment"] == microservice].copy()

        if data.empty:
            logger(f"No {microservice} data for {label}", level="WARNING")
            continue

        if "relative_min" not in data.columns:
            raise KeyError(f"Column 'relative_min' not found for {label}.")

        if replica_col not in data.columns:
            raise KeyError(f"Column '{replica_col}' not found for {label}.")

        data = data.sort_values("relative_min").reset_index(drop=True)
        data["relative_h"] = data["relative_min"] / 60.0

        series[label] = data[["relative_min", "relative_h", replica_col]].dropna().copy()

    return series
    
def plot_frontend_replicas_zoom(
    prometheus_datasets,
    labels,
    file_name,
    xlim,
    which_replika="available",
    title=None,
    microservice="frontend"
):
    replica_col = "available_replicas" if which_replika == "available" else "desired_replicas"
    series = get_microservice_series(prometheus_datasets, labels, replica_col=replica_col, microservice=microservice)

    plt.figure(figsize=(12, 5))

    y_min = None
    y_max = None

    for label, data in series.items():
        part = data[(data["relative_h"] >= xlim[0]) & (data["relative_h"] <= xlim[1])].copy()
        if part.empty:
            logger(f"No data in zoom interval {xlim} for {label}", level="WARNING")
            continue

        lw = 2.2 if "Hybrid" in label else 2.0
        plt.step(
            part["relative_h"],
            part[replica_col],
            where="post",
            linewidth=lw,
            label=label
        )

        cur_min = part[replica_col].min()
        cur_max = part[replica_col].max()
        y_min = cur_min if y_min is None else min(y_min, cur_min)
        y_max = cur_max if y_max is None else max(y_max, cur_max)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    plt.grid(True, alpha=0.25)

    plt.xlim(*xlim)
    if y_min is not None and y_max is not None:
        plt.ylim(max(0, y_min - 0.5), y_max + 0.5)

    plt.xlabel("Čas od začiatku experimentu (hodiny)")
    plt.ylabel("Počet replík")
    if title:
        plt.title(title)

    legend = plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        borderaxespad=0,
        title="Technika škálovania"
    )
    # legend.get_title().set_weight("bold")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_frontend_replicas_difference(
    prometheus_datasets,
    labels,
    baseline_label="HPA",
    file_name="frontend_available_diff_vs_hpa",
    which_replika="available",
    title=None
):
    replica_col = "available_replicas" if which_replika == "available" else "desired_replicas"
    series = get_microservice_series(prometheus_datasets, labels, replica_col=replica_col)

    if baseline_label not in series:
        raise ValueError(f"Baseline label '{baseline_label}' not found in datasets. Found: {list(series.keys())}")

    base = series[baseline_label].copy()
    base = base.rename(columns={replica_col: "baseline_value"})

    plt.figure(figsize=(12, 5))

    plotted_any = False

    for label, data in series.items():
        if label == baseline_label:
            continue

        merged = pd.merge_asof(
            data.sort_values("relative_h"),
            base.sort_values("relative_h"),
            on="relative_h",
            direction="nearest",
            tolerance=1/120  # 30 sekúnd = 0.00833h
        ).dropna()

        if merged.empty:
            logger(f"No aligned data for difference plot: {label} vs {baseline_label}", level="WARNING")
            continue

        merged["diff"] = merged[replica_col] - merged["baseline_value"]

        plt.step(
            merged["relative_h"],
            merged["diff"],
            where="post",
            linewidth=2.2,
            label=f"{label} - {baseline_label}"
        )
        plotted_any = True

    plt.axhline(0, linestyle="--", linewidth=1)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.grid(True, alpha=0.25)

    plt.xlabel("Čas od začiatku experimentu (hodiny)")
    plt.ylabel("Rozdiel v počte replík")
    if title:
        plt.title(title)

    if plotted_any:
        legend = plt.legend(
            # loc="upper left",
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            title="Technika škálovania"
        )
        # legend.get_title().set_weight("bold")

    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_frontend_replicas_zoom_min(
    prometheus_datasets,
    labels,
    file_name,
    xlim,
    which_replika="available",
    legend_loc="upper left",
    title=None,
    microservice="frontend"
):
    replica_col = "available_replicas" if which_replika == "available" else "desired_replicas"
    series = get_microservice_series(prometheus_datasets, labels, replica_col=replica_col, microservice=microservice)

    plt.figure(figsize=(12, 5))

    y_min = None
    y_max = None

    for label, data in series.items():
        # 🔥 použijeme relative_min namiesto hodín
        part = data[(data["relative_min"] >= xlim[0]) & (data["relative_min"] <= xlim[1])].copy()

        if part.empty:
            logger(f"No data in zoom interval {xlim} for {label}", level="WARNING")
            continue

        lw = 2.2 if "Hybrid" in label else 2.0

        plt.step(
            part["relative_min"],
            part[replica_col],
            where="post",
            linewidth=lw,
            label=label
        )

        cur_min = part[replica_col].min()
        cur_max = part[replica_col].max()
        y_min = cur_min if y_min is None else min(y_min, cur_min)
        y_max = cur_max if y_max is None else max(y_max, cur_max)

    ax = plt.gca()

    # 🔥 tick každých 10 minút (môžeš doladiť)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    plt.grid(True, alpha=0.25)

    plt.xlim(*xlim)

    if y_min is not None and y_max is not None:
        plt.ylim(max(0, y_min - 0.5), y_max + 0.5)

    plt.xlabel("Čas od začiatku experimentu (minúty)")
    plt.ylabel("Počet replík")

    if title:
        plt.title(title)

    legend = plt.legend(
        loc=legend_loc,
        # bbox_to_anchor=(1, 1),
        # borderaxespad=0,
        title="Technika škálovania",
        frameon=True,
    )
    # legend.get_title().set_weight("bold")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------- 
# Dataset saving functions
def split_dataset_by_day(dataset, days_to_split = 7, save=True):
    if save and not os.path.exists(SYNTHETIC_YEAR_DIR):
        os.makedirs(SYNTHETIC_YEAR_DIR)
    
    start_timestamp = START_TIMESTAMP  # Corresponds to 2024-01-01 00:00:00
    day_seconds = 24 * 60 * 60 # Number of seconds in a day
    dataset_list = []

    for x in range(days_to_split):
        starter = (x * day_seconds) + start_timestamp
        ender = starter + day_seconds

        day_dataset = dataset[(dataset["timestamp"] >= starter) & (dataset["timestamp"] < ender)].copy()
        day_dataset = day_dataset.sort_values(['timestamp', 'deployment']).reset_index(drop=True)
        dataset_list.append(day_dataset)

        if save:
            file_name = datetime.fromtimestamp(starter).strftime("%Y%m%d.csv")
            output_file_path = os.path.join(SYNTHETIC_YEAR_DIR, file_name)
            day_dataset.to_csv(output_file_path, index=False)
            logger(f"Saved Dataset: {file_name} with {len(day_dataset)} rows.")

    return dataset_list