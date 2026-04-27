import os
import yaml
import time

from datetime import datetime, timedelta
from my_modules.prometheus_connector import load_data_for_prediction
from my_modules.model_utils import load_model
from my_modules.printable_utils import logs

def main():
    '''
    main function for predicting future classes
    '''

    # load metadata/paths to dirs
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        metadata = config["metadata"]
        model_name = "mlp_oneStep"

        model_dir = os.path.join(BASE, config["paths"][model_name]["model_path"])
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        history_path = os.path.join(model_dir, "history_data.pkl")

        sleep_time = metadata["model_param"]["prediction_sleep"]
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # Track last modified time of the model
    last_model_mtime, clf, scaler = None, None, None

    while True:
        # Check if the model file has changed
        if os.path.exists(model_path):
            current_mtime = os.path.getmtime(model_path)
            if last_model_mtime is None or current_mtime > last_model_mtime:
                load_result = load_model(model_path=model_path, scaler_path=scaler_path, history_path=history_path)
                if load_result is not None:
                    clf, scaler, _, _ = load_result
                    logs(f"Model reloaded at {datetime.now()}")
                last_model_mtime = current_mtime

        if clf is None or scaler is None:
            logs("No model loaded yet, skipping prediction")
        else:
            # MLP Training on kubernetes data from prometheus
            skipped_steps = metadata["data_param"]["skipped_steps"]
            feature_multiplier = metadata["data_param"]["features_multiplier"] - 2
            load_number = feature_multiplier + skipped_steps
            end_time = datetime.now().replace(second=0, microsecond=0)
            start_time = end_time - timedelta(minutes=load_number)
            # load data, deployment names and timestamps
            data, deployment_names, timestamps = load_data_for_prediction(metadata, start_time, end_time)

            # if data loaded
            if data.size > 0:
                preds = clf.predict(scaler.transform(data))
                for ts, dep, pred in zip(timestamps, deployment_names, preds):
                    future_ts_unix = int(ts + skipped_steps * 60)
                    print(datetime.fromtimestamp(future_ts_unix), f"Deployment: {dep}, Predicted class: {pred}")
            else:
                logs("No data available for prediction")

        time.sleep(sleep_time)  # run every minute


if __name__ == '__main__':
    main()
