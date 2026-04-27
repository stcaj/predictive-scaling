from concurrent.futures import ThreadPoolExecutor
import os
import traceback
import numpy as np
import yaml

from datetime import datetime, timedelta
from my_modules.kubernetes_connector import kuber_replica_class_autoscale
from my_modules.prometheus_connector import load_current_classes, load_data_for_prediction
from my_modules.model_utils import load_model, wait_some_min
from my_modules.printable_utils import init_logs, logs


def main():
    '''
    main function for predicting future classes
    '''
    # load metadata/paths to dirs
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config-scaling.yaml')) as scaling_file:
            scaling_config = yaml.safe_load(scaling_file)

        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        # metadata
        metadata = config["metadata"]
        paths = config["paths"]
        # model_name = "mlp_multiStep"
        # input_type = "cpu_features"


        model_name = metadata["data_param"]["model_name"]
        input_type = metadata["data_param"]["input_type"]
        strategy = metadata["data_param"]["strategy"]

        UP_CLASS = metadata["data_param"]["UP_CLASS"]
        DOWN_CLASS = metadata["data_param"]["DOWN_CLASS"]
        TARGET_CLASS = metadata["data_param"]["TARGET_CLASS"]

        scaling_deployments = set(metadata["scale_deployments"])

        # model path
        model_dir = os.path.join(BASE, paths[model_name]["model_path"])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        history_path = os.path.join(model_dir, "history_data.pkl")

        # MLP Training on kubernetes data from prometheus
        # predicted_steps = metadata["model_param"]["mlp_multiStep_future_labels"]
        feature_multiplier = metadata["data_param"]["features_multiplier"]
        load_number = feature_multiplier

        # timer
        # SLEEP_TIME = metadata["model_param"]["prediction_sleep_minutes"]
        SLEEP_TIME = int(os.getenv("SLEEP_TIME", metadata["model_param"]["prediction_sleep_minutes"]))
        WARMUP_MINUTES = int(os.getenv("WARMUP_MINUTES", "2"))

        # initiate logger
        # init_logs(log_file=paths["logs"]["prediction_log_path"])
        log_file_path = os.getenv("LOG_FILE", paths["logs"]["prediction_log_path"])
        init_logs(log_file=log_file_path)

    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # Track last modified time of the model
    last_model_mtime, clf, scaler = None, None, None

    # ------------------------------------------------------------------------ fix time 
    # pred while True:
    wait_some_min(WARMUP_MINUTES)

    while True:
        logs("----------")
        try:            
            # Check if the model file has changed
            if os.path.exists(model_path):
                current_mtime = os.path.getmtime(model_path)
                if last_model_mtime is None or current_mtime > last_model_mtime:
                    logs(f"Loading new model")
                    load_result = load_model(model_path=model_path, scaler_path=scaler_path, history_path=history_path)
                    if load_result is not None:
                        clf, scaler, _, _ = load_result
                        logs(f"Model reloaded succesfully")
                    last_model_mtime = current_mtime            
            else:
                logs("No model loaded yet -> skipping prediction")

            # continue if model is available
            if clf is not None and scaler is not None:
                end_time = datetime.now().replace(second=0, microsecond=0)
                start_time = end_time - timedelta(minutes=load_number)
                logs(f"Loading data from {start_time} to {end_time}")
                # load data, deployment names and timestamps
                # data, deployment_names, _ = load_data_for_prediction(metadata, start_time, end_time, anchor_time)
                data, deployment_names, _ = load_data_for_prediction(
                                                metadata=metadata,
                                                start_time=start_time,
                                                end_time=end_time,
                                                input_type=input_type,
                                            )
                logs("Data loaded succesfully")
                # logs(f"Data:\n{data}")
                logs(f"Raw deployments [{len(deployment_names)}] from Prometheus: {set(deployment_names)}")

                # if data loaded
                if data.size > 0:
                    n_deps = len(metadata["app_deployments"])
                    n_numeric = data.shape[1] - n_deps

                    X_num = data[:, :n_numeric].astype(float, copy=False)
                    X_dep = data[:, n_numeric:].astype(float, copy=False)

                    X_num = scaler.transform(X_num)
                    X = np.hstack([X_num, X_dep])

                    predictions = clf.predict(X)
                    current_cpus = load_current_classes(
                        metadata=metadata,
                        end_time=end_time,
                        start_time=start_time
                    )

                    logs(f"Predicted deployments: {deployment_names}")
                    logs(f"Allowed scaling deployments: {sorted(scaling_deployments)}")

                    deployments_to_scale = [
                        (dep, pred)
                        for dep, pred in zip(deployment_names, predictions)
                        if dep in scaling_deployments
                    ]

                    if deployments_to_scale:
                        logs(f"Deployments selected for scaling: {[dep for dep, _ in deployments_to_scale]}")

                        with ThreadPoolExecutor(max_workers=len(deployments_to_scale)) as executor:
                            futures = [
                                executor.submit(
                                    kuber_replica_class_autoscale,
                                    deployment_name=dep,
                                    predicted_cpu_classes=pred,
                                    current_class=current_cpus.get(dep, {}).get("class", 0.0),
                                    scaling_config=scaling_config,
                                    UP_CLASS=UP_CLASS,
                                    DOWN_CLASS=DOWN_CLASS,
                                    TARGET_CLASS=TARGET_CLASS,
                                    strategy=strategy,
                                )
                                for dep, pred in deployments_to_scale
                            ]

                            for future in futures:
                                future.result()
                    else:
                        logs("No deployments matched scale_deployments -> skipping autoscaling")
                else:
                    logs("No data available for prediction")

        except Exception as e:
            logs(f"Error inside training loop...")
            logs(f"Exception: {e}")
            logs(traceback.format_exc())

        finally:    
            wait_some_min(SLEEP_TIME)


if __name__ == '__main__':
    main()
