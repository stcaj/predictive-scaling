# prometheus_collector_daily.py

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yaml

from my_modules.printable_utils import logs
from my_modules.prometheus_connector import load_microservice_data


# --- KONFIGURÁCIA ---
BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, 'config.yaml')) as file:
    config = yaml.safe_load(file)

PROMETHEUS_URL = config["prometheus_config"]["url"]
deployments = config["app_config"]["collection_deployments"]

STEP_SECONDS = config["lstm_config"]["metadata"]["data_param"]["step_seconds"]

OUTPUT_DIR = os.path.join(BASE, "logs/prometheus")
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def append_to_daily_csv(df: pd.DataFrame, base_dir: str):
    """Rozdelí DF podľa dňa (podľa timestamp stĺpca)"""
    if df.empty:
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

    for date_value, group in df.groupby("date"):
        filename = os.path.join(base_dir, f"metrics_%s.csv" % date_value)

        group = group.drop(columns=["date"])
        file_exists = os.path.isfile(filename)
        group.to_csv(
            filename,
            mode="a",
            index=False,
            header=not file_exists,
        )
        logs(f"Appended {len(group)} rows to {filename}.")


def main():
    ensure_output_dir()
    logs("Starting Prometheus daily collector...")
    now = datetime.now()
    now_floor = now.replace(second=0, microsecond=0)
    last_end_time = now_floor - timedelta(seconds=STEP_SECONDS)

    try:
        while True:
            end_time = datetime.now().replace(second=0, microsecond=0)
            if (end_time - last_end_time).total_seconds() < STEP_SECONDS:
                time.sleep(1)
                continue

            start_time = last_end_time
            logs(f"Collecting metrics from {start_time} to {end_time} ...")

            df = load_microservice_data(
                deployments=deployments,
                PROMETHEUS_URL=PROMETHEUS_URL,
                start_time=start_time,
                end_time=end_time,
                step=STEP_SECONDS,
            )

            if not df.empty:
                append_to_daily_csv(df, OUTPUT_DIR)
            else:
                logs("No data returned for this interval.")

            last_end_time = end_time
            time.sleep(STEP_SECONDS)

    except KeyboardInterrupt:
        logs("Collector stopped by user.")


if __name__ == "__main__":
    main()
