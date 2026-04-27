import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import logger, save_dataset
from sdv.io.local import CSVHandler
from tsaug import AddNoise, Drift

# DIRS
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_preprocessing')
SYNTHETIC_YEAR_DIR = os.path.join(BASE_DIR, 'synthetic_year')

# FILES
# SYNTHETIC_WEEK_FILE = os.path.join(DATA_DIR, "synthetic_week_imputed.csv")
SYNTHETIC_WEEK_FILE = os.path.join(DATA_DIR, "synthetic_week_final.csv")

# CONSTANTS
TIME_ZONE = "Europe/Bratislava"
WEEK_ADDON = 7 * 24 * 60 * 60
NUM_OF_SYNTHETIC_WEEKS = 53
# NUM_OF_SYNTHETIC_WEEKS = 3

ATTRIBUTES = [
    "timestamp",
    "hour", "minute",
    "day_of_week",
    "month",
    "deployment",
    "cpu_percent",
    "memory_percent",
    "desired_replicas", "available_replicas"
]

# Augmentačné parametre
AUGMENTED_COLS = ("cpu_percent", "memory_percent")
# ------------------------------------------------------------------------------------------------------------------------
def shift_time_columns(df: pd.DataFrame):
    dt = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert(TIME_ZONE)
    )
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    return df


# ------------------------------------------------------------------------------------------------------------------------
def compute_stats(weeks, who_for="dataset", show=True):
    if isinstance(weeks, pd.DataFrame):
        all_data = weeks
    else:
        all_data = pd.concat(weeks, ignore_index=True)

    stats = (
        all_data
        .groupby("deployment")["cpu_percent"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            p95=lambda x: x.quantile(0.95)
        )
    )

    if show:
        logger(
            f"Štatistiky CPU percent pre jednotlivé deploymenty ({who_for}):\n{stats}",
            level="info"
        )

    return stats

def compute_stats_for_selected_weeks(weeks, week_numbers, deployment_filter=None):
    results = {}
    for w in week_numbers:
        idx = w - 1  # lebo zoznam indexuje od 0

        if idx >= len(weeks):
            print(f"Týždeň {w} neexistuje.")
            continue

        df = weeks[idx]

        if deployment_filter:
            df = df[df["deployment"] == deployment_filter]

        stats = (
            df
            .groupby("deployment")["cpu_percent"]
            .agg(
                mean="mean",
                std="std",
                p95=lambda x: x.quantile(0.95)
            )
        )

        results[w] = stats

    return results
# ------------------------------------------------------------------------------------------------------------------------
def get_deployment_augmentation_config(dep: str):
    
    p95_real = REAL_TARGET[dep]["p95"]
    intensity = dep_intensity(p95_real)

    mean_real = REAL_TARGET[dep]["mean"]
    mean_factor = float(np.clip(mean_real / 10.0, 0.15, 1.0))

    # menšie služby (nízka intensity) -> výrazne menej noise/drift
    noise_scale = (0.02 + 0.05 * intensity) * mean_factor
    drift_max   = (0.04 + 0.12 * intensity) * mean_factor
    
    return (Drift(max_drift=drift_max)
            + AddNoise(scale=noise_scale)
        )


# ------------------------------------------------------------------------------------------------------------------------
def dep_intensity(p95):
    return float(np.clip(p95 / 95.0, 0.15, 1.0))

# ------------------------------------------------------------------------------------------------------------------------
# def dep_cap(dep):
#     p95 = REAL_TARGET[dep]["p95"]
#     mx  = REAL_TARGET[dep]["max"]
#     # cap
#     # cap = max(p95 * 1.08, mx * 1.01)
#     cap = min(99.0, max(mx, p95 * 1.08))
#     return float(min(99.0, cap))
def dep_cap(dep):
    p95 = REAL_TARGET[dep]["p95"]
    mx = REAL_TARGET[dep]["max"]
    return float(max(mx * 1.05, p95 * 1.15))

# ------------------------------------------------------------------------------------------------------------------------
def soft_cap(x, cap, softness=1.5):
    x = np.asarray(x, dtype=np.float32)
    over = x - cap
    return np.where(
        over <= 0.0,
        x,
        cap + softness * np.tanh(over / softness)
    )

# ------------------------------------------------------------------------------------------------------------------------
def apply_constraints(X_aug, dep: str):
    cap = dep_cap(dep)

    cpu_raw = X_aug[:, 0]

    real_min = float(REAL_STATS.loc[dep, "min"])
    floor = 0.0 if real_min == 0.0 else 0.8 * real_min

    cpu = np.maximum(cpu_raw, floor)
    cpu = soft_cap(cpu, cap, softness=3.0)
    cpu = np.clip(cpu, floor, cap)

    mem = np.clip(X_aug[:, 1], 0.0, 100.0)
    return cpu, mem
# ------------------------------------------------------------------------------------------------------------------------
def augment_week(base_week_df: pd.DataFrame, week: int):
    df = base_week_df.copy()
    df["timestamp"] = df["timestamp"] + week * WEEK_ADDON
    df = shift_time_columns(df)

    out = df.copy()
    deployments = sorted(out["deployment"].unique())

    for dep in deployments:
        g = out[out["deployment"] == dep].sort_values("timestamp").reset_index()
        X = g[list(AUGMENTED_COLS)].to_numpy(dtype=np.float32)
        aug = get_deployment_augmentation_config(dep)
        X_aug = aug.augment(X)

        cpu, mem = apply_constraints(X_aug, dep)

        idx = g["index"].to_numpy()
        out.loc[idx, "cpu_percent"] = cpu
        out.loc[idx, "memory_percent"] = mem

    return out

# ------------------------------------------------------------------------------------------------------------------------
def synthesize_and_save_dataset():
    logger("Starting dataset synthesis...", level="info")
    os.makedirs(SYNTHETIC_YEAR_DIR, exist_ok=True)

    base_week_df = load_dataset(SYNTHETIC_WEEK_FILE)
    if base_week_df is None:
        logger("Failed to load dataset.", level="error")
        return None

    base_week_df = keep_attributes(base_week_df, ATTRIBUTES)

    counts = base_week_df["deployment"].value_counts()
    if counts.min() != counts.max():
        logger(f"Warning: deployments majú rôzne počty riadkov: {counts.to_dict()}", level="warning")

    # synthesize and save each week
    synthetic_weeks = []
    for week in tqdm(range(NUM_OF_SYNTHETIC_WEEKS)):
        week_df = augment_week(base_week_df=base_week_df, week=week)
        file_name = f"synthetic_week_{week+1:02d}.csv"
        file_path = os.path.join(SYNTHETIC_YEAR_DIR, file_name)
        save_dataset(week_df, file_path, debug=False)
        synthetic_weeks.append(week_df)

    return synthetic_weeks

# ------------------------------------------------------------------------------------------------------------------------
def show_weekly_plots(
    weeks,
    attribute_to_plot="cpu_percent",
    weeks_to_compare=(1, 13, 26, 39, 52),
    number_of_minutes_to_plot=7 * 24 * 60,
):
    logger(
        f"Showing weekly plots for {len(weeks_to_compare)} weeks, attribute: {attribute_to_plot}",
        level="info",
    )

    # vyber presne požadované týždne podľa weeks_to_compare
    selected = []
    for w in weeks_to_compare:
        idx = w - 1
        if 0 <= idx < len(weeks):
            selected.append((w, weeks[idx]))
        else:
            logger(f"Week {w} not available (len(weeks)={len(weeks)})", level="warning")

    if not selected:
        logger("No weeks selected — nothing to plot.", level="warning")
        return

    # Pre každý deployment sprav vlastný graf (aby sa nemiešali)
    for deployment in selected[0][1]["deployment"].unique():
        plt.figure(figsize=(14, 6))

        for w, week_df in selected:
            dep_data = (
                week_df[week_df["deployment"] == deployment]
                .sort_values("timestamp")
                .copy()
            )

            # relatívny čas v minútach (ak timestamp je v sekundách)
            dep_data["t_minutes"] = (dep_data["timestamp"] - dep_data["timestamp"].min()) / 60.0

            # orezanie podľa počtu minút
            dep_data = dep_data[dep_data["t_minutes"] <= number_of_minutes_to_plot]

            plt.plot(
                dep_data["t_minutes"],
                dep_data[attribute_to_plot],
                linewidth=1.2,
                alpha=0.8,
                label=f"Week {w}",
            )

        plt.title(f"Comparison of weeks {list(weeks_to_compare)} ({deployment})")
        plt.xlabel("Time (minutes from start of week)")
        plt.ylabel(attribute_to_plot)
        plt.legend()
        plt.tight_layout()
        plt.show()
# ------------------------------------------------------------------------------------------------------------------------
def show_real_vs_synthetic(dataset=None):
        compute_stats(REAL_DATA, who_for="real")
        compute_stats(dataset, who_for="synthetic")
        selected_weeks = [1, 13, 26, 39, 52]

        results = compute_stats_for_selected_weeks(
            dataset,
            selected_weeks,
            deployment_filter="frontend"  # môžeš dať None pre všetky
        )

        for week, stats in results.items():
            print(f"\n===== Week {week} =====")
            print(stats)

# ------------------------------------------------------------------------------------------------------------------------
def keep_attributes(df, attributes):
    attributes = [attr for attr in attributes if attr in df.columns]
    df_after = df[attributes].copy()
    # logger(f"Kept attributes: {attributes}. Dropped others.", level="info")
    return df_after

# ------------------------------------------------------------------------------------------------------------------------
def load_dataset(file_path, debug=True):
    try:
        connector = CSVHandler()
        file_name = os.path.basename(file_path)
        data = connector.read(
            folder_name=os.path.dirname(file_path),
            file_names=[file_name],
            read_csv_parameters={'parse_dates': False, 'encoding': 'latin-1'}
        )
        if debug:
            logger(f"Dataset loaded from {file_path}", level="info")
        return data[file_name.replace('.csv', '')]
    except Exception as e:
        logger(f"Failed to load dataset from {file_path}: {e}", level="error")
        return None
# ------------------------------------------------------------------------------------------------------------------------
def load_all():
    synthetic_weeks = []
    for week in tqdm(range(NUM_OF_SYNTHETIC_WEEKS)):
        file_name = f"synthetic_week_{week+1:02d}.csv"
        file_path = os.path.join(SYNTHETIC_YEAR_DIR, file_name)
        df = load_dataset(file_path, debug=False)
        if df is not None:
            synthetic_weeks.append(df)
    return synthetic_weeks
# ------------------------------------------------------------------------------------------------------------------------
def main():
    try:
        dataset = None
        dataset = synthesize_and_save_dataset()

        # načítaj týždne zo súborov, ak už boli vygenerované a v tomto skripte je zakomentované generovanie
        if dataset is None:
            logger(f"Loading synthetic weeks.", level="info")
            dataset = load_all()
        show_real_vs_synthetic(dataset)
        show_weekly_plots(dataset)
    except Exception as e:
        logger(f"Exception:{e}", level="error")


REAL_DATA = load_dataset(SYNTHETIC_WEEK_FILE)
REAL_STATS = compute_stats(REAL_DATA, who_for="real", show=False)
REAL_TARGET = REAL_STATS[["mean", "p95", "max"]].to_dict("index")
if __name__ == "__main__":
    main()
