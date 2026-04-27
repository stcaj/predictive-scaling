from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import os
import traceback
import yaml
import numpy as np

from my_modules.kubernetes_connector import kuber_replica_autoscale
from my_modules.prometheus_connector import load_data_for_prediction
from my_modules.model_utils import load_model, wait_some_min
from my_modules.printable_utils import init_logs, logs


def load_microservice_model(model_path, model_name=""):
    """
    Load one microservice model and its scaler/normalizer.
    Expected return from load_model():
        (model, scaler, normalize_scaler)
    """
    if os.path.exists(model_path):
        load_result = load_model(model_name=model_name, data_path=model_path)
        if load_result:
            model, scaler, normalize_scaler = load_result
            return model, scaler, normalize_scaler
        logs(f"Model exists but failed to load: {model_path}")
        return None

    logs(f"No model found at path: {model_path}")
    return None


def load_models(model_dir, deployments, model_name):
    """
    Load models for all deployments.

    Returns:
        model_dict, scaler_dict, normalizer_dict
    """
    model_dict = {}
    scaler_dict = {}
    normalizer_dict = {}

    for deployment in deployments:
        deployment_model_path = os.path.join(model_dir, deployment)
        correct_model_name = f"{deployment}_{model_name}_single_feature"
        load_result = load_microservice_model(
            model_path=deployment_model_path,
            model_name=correct_model_name
        )

        if load_result is not None:
            model, scaler, normalize_scaler = load_result
            model_dict[deployment] = model
            scaler_dict[deployment] = scaler
            normalizer_dict[deployment] = normalize_scaler
            logs(f"Loaded model for deployment: {deployment}")
        else:
            logs(f"Could not load model for deployment: {deployment}")

    if not model_dict:
        return None, None, None

    return model_dict, scaler_dict, normalizer_dict


def prepare_scaled_sequences(X_raw: np.ndarray, scaler):
    """
    Scale raw 3D sequences using a fitted scaler.
    Input shape: (samples, seq_len, features)
    """
    if X_raw is None or len(X_raw) == 0:
        return None

    n_samples, seq_len, n_features = X_raw.shape
    X_2d = X_raw.reshape(-1, n_features)
    X_scaled_2d = scaler.transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)

    return np.asarray(X_scaled, dtype=np.float32)


def predict_for_deployment(deploy, prediction_data, model_dict, scaler_dict, normalizer_dict):
    if deploy not in prediction_data:
        logs(f"No prediction data for deployment: {deploy}")
        return deploy, None, None

    if deploy not in model_dict or deploy not in scaler_dict or deploy not in normalizer_dict:
        logs(f"Missing model/scaler/normalizer for deployment: {deploy}")
        return deploy, None, None

    X_raw = prediction_data[deploy]["X_raw"]

    if X_raw is None or len(X_raw) == 0:
        logs(f"Empty X_raw for deployment: {deploy}")
        return deploy, None, None

    # posledná aktuálna CPU hodnota z poslednej sekvencie
    current_cpu_percent = float(X_raw[-1, -1, 0])
    # current_cpu_percent = float(np.mean(X_raw[-1, -3:, 0]))

    X_scaled = prepare_scaled_sequences(X_raw, scaler_dict[deploy])
    
    if X_scaled is None:
        logs(f"Scaling failed for deployment: {deploy}")
        return deploy, None, None

    y_pred_scaled = model_dict[deploy].predict(X_scaled, verbose=0)

    pred_shape = y_pred_scaled.shape
    y_pred = normalizer_dict[deploy].inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(pred_shape)
    y_pred = np.clip(y_pred, 0, 100)
    
    last_pred = y_pred[-1]
    predicted_cpu_percentages = np.asarray(last_pred).reshape(-1).tolist()

    logs(
        f"[PRED] deployment={deploy}, "
        f"X_raw={X_raw.shape}, "
        f"X_scaled={X_scaled.shape}, "
        f"y_pred_scaled_shape={y_pred_scaled.shape}, "
        f"current_cpu={current_cpu_percent}, "
        f"pred={predicted_cpu_percentages}"
    )

    # logs(f"Scaler min: {normalizer_dict[deploy].data_min_}")
    # logs(f"Scaler max: {normalizer_dict[deploy].data_max_}")

    return deploy, predicted_cpu_percentages, current_cpu_percent


def main():
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(BASE, "config-scaling.yaml"), "r", encoding="utf-8") as file:
            scaling_config = yaml.safe_load(file)
        service_scaling = scaling_config["services"]

        with open(os.path.join(BASE, "config.yaml"), "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        model_name = os.getenv("MODEL")
        if not model_name:
            raise ValueError("Environment variable MODEL is not set.")

        metadata = config[f"{model_name}_config"]["metadata"]
        model_param = metadata["model_param"]
        seq_length = model_param.get("seq_length", 128)

        # this is optional; kept here only if you later want dynamic feature selection
        input_features = config["prometheus_config"].get("multi_features", ["cpu_percent"])

        # prediction metadata
        future_steps = config["prediction_config"]["future_steps"]
        strategy = config["prediction_config"]["strategy"]
        pod_cpu_target = config["prediction_config"]["pod_cpu_target"]
        upscale_threshold = config["prediction_config"]["upscale_threshold"]
        downscale_threshold = config["prediction_config"]["downscale_threshold"]
        upscale_cooldown_seconds=config["prediction_config"]["upscale_cooldown_seconds"]
        downscale_cooldown_seconds=config["prediction_config"]["downscale_cooldown_seconds"]
        downscale_after_upscale_delay_seconds = config["prediction_config"]["downscale_after_upscale_delay_seconds"]
        upscale_replica_limit = config["prediction_config"]["upscale_replica_limit"]
        downscale_replica_limit = config["prediction_config"]["downscale_replica_limit"]

        # prometheus + deployments
        PROMETHEUS_URL = config["prometheus_config"]["url"]
        deployments = config["app_config"]["deployments"]

        # model paths
        save_dir = os.getenv("SAVE_DIR", config[f"{model_name}_config"]["paths"]["save_dir"])
        abs_save_dir = os.path.join(BASE, save_dir)
        model_dir = os.path.join(abs_save_dir, "model")

        # timer
        sleep_time = int(os.getenv("SLEEP_TIME", metadata["data_param"]["prediction_sleep"]))
        step_time = metadata["data_param"]["step_seconds"]

        # logger
        log_file_path = os.getenv("LOG_FILE", config[f"{model_name}_config"]["paths"]["prediction_logs"])
        init_logs(log_file=log_file_path)

        logs(f"MODEL={model_name}")
        logs(f"SEQ_LENGTH={seq_length}")
        logs(f"STEP_TIME={step_time}")
        logs(f"MODEL_DIR={model_dir}")
        logs(f"Deployments={deployments}")
        logs(f"Future steps={future_steps}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        print(traceback.format_exc())
        raise SystemExit(-1)

    model_dict, scaler_dict, normalizer_dict = load_models(
        model_dir=model_dir,
        deployments=deployments,
        model_name=model_name
    )

    if model_dict is None:
        logs("Error loading models.")
        raise SystemExit(-1)

    while True:
        logs("----------")

        try:
            end_time = datetime.now().replace(second=0, microsecond=0)
            start_time = end_time - timedelta(seconds=seq_length * step_time)

            logs(f"Loading data from {start_time} to {end_time}")

            prediction_data = load_data_for_prediction(
                metadata=metadata,
                PROMETHEUS_URL=PROMETHEUS_URL,
                deployments=deployments,
                start_time=start_time,
                end_time=end_time,
                seq_length=seq_length,
                feature_columns=["cpu_percent"]
            )

            if not prediction_data:
                logs("No data available for prediction.")
                wait_some_min(sleep_time)
                continue

            all_predictions = {}

            for deploy in deployments:
                dep, preds, current_cpu = predict_for_deployment(
                    deploy=deploy,
                    prediction_data=prediction_data,
                    model_dict=model_dict,
                    scaler_dict=scaler_dict,
                    normalizer_dict=normalizer_dict
                )

                if preds is not None:
                    all_predictions[dep] = {
                        "preds": preds[:future_steps],
                        "current_cpu": current_cpu
                    }

            if not all_predictions:
                logs("No predictions available for autoscaling.")
                wait_some_min(sleep_time)
                continue

            max_workers = max(1, len(all_predictions))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        kuber_replica_autoscale,
                        deployment_name=dep,
                        predicted_cpu_percentages=data["preds"],
                        current_cpu_percent=data["current_cpu"],
                        min_replicas=service_scaling[dep]["min_pods"],
                        max_replicas=service_scaling[dep]["max_pods"],
                        strategy=strategy,
                        target_cpu_percent=pod_cpu_target,
                        upscale_threshold=upscale_threshold,
                        downscale_threshold=downscale_threshold,
                        upscale_cooldown_seconds=upscale_cooldown_seconds,
                        downscale_cooldown_seconds=downscale_cooldown_seconds,
                        downscale_after_upscale_delay_seconds=downscale_after_upscale_delay_seconds,
                        upscale_replica_limit=upscale_replica_limit,
                        downscale_replica_limit=downscale_replica_limit,
                    ): dep
                    for dep, data in all_predictions.items()
                }

                for future in futures:
                    dep = futures[future]
                    try:
                        future.result()
                        logs(f"Autoscale finished for deployment: {dep}")
                    except Exception as e:
                        logs(f"Autoscale failed for deployment {dep}: {e}")
                        logs(traceback.format_exc())

        except Exception as e:
            logs(f"Prediction loop error: {e}")
            logs(traceback.format_exc())

        finally:
            wait_some_min(sleep_time)


if __name__ == "__main__":
    main()
