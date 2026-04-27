import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import traceback
import numpy as np
import tqdm
import keras
import yaml

from datetime import datetime
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from my_modules.prometheus_connector import preprocess_data, preprocess_data_multi
from my_modules.data_util import create_sequences_only, data_split, data_split_multi
from my_modules.model_utils import compare_models, evaluate_final_model, evaluate_model, get_callbacks, load_model, save_model, wait_for_midnight
from my_modules.printable_utils import create_graphs, init_console, init_logs, logs
from my_modules.offline_testing_utils import compute_epoch_durations, load_synthetic_dataset, split_dataset_by_interval

from main_lstm_multi import build_lstm_multi_model
from main_hybrid_multi import build_hybrid_multi_model


def get_model(
        model_name: str, 
        seq_length, 
        feature_count = None, 
        units = None, 
        units1=None, 
        units2=None
    ):
    if model_name == 'lstm':        return build_lstm_multi_model(seq_length=seq_length, feature_count=feature_count, units1=units1, units2=units2)
    elif model_name == 'hybrid':    return build_hybrid_multi_model(seq_length=seq_length, feature_count=feature_count, units=units)
    else: 
        raise ValueError(f"{model_name} is wrong...")


# MODEL_NAMES = ["lstm",'hybrid']
MODEL_NAMES = ['hybrid']
# INPUT_FEATURES = ["single_feature","multi_features","multi_features_mem"]
INPUT_FEATURES = ["single_feature"]
TARGET = "cpu_percent"
INTERVALS = [
    # 1, # 1 day -> 364 times (every day)
    # 7,  # 7 days -> 52 times (every week)
    # 31, # 31 days -> 12 times (every month)
    364, # 364 days -> 1 time (every year) 
]

def train(train_dataset):
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        DEPLOYMENTS = config["app_config"]["deployments"]
        TEST_DIR_PATH = os.path.join(BASE, "offline_tests_out")

    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    for deployment in tqdm.tqdm(DEPLOYMENTS, desc="Deployments", leave=False):
        service_dataset = train_dataset[train_dataset["deployment"] == deployment].copy()

        if service_dataset.empty:
            logs(f"No data for deployment={deployment} -> skipping.")
            continue

        for model_name in tqdm.tqdm(MODEL_NAMES, desc=f"Training: {deployment}", leave=False):
            metadata = config[f"{model_name}_config"]["metadata"]
            model_param = metadata["model_param"]
            future_steps = config["prediction_config"]["future_steps"]

            EPOCHS = int(os.getenv("EPOCHS", model_param.get("epochs", 10)))
            seq_length = model_param.get("seq_length", 128)
            batch_size = model_param.get("batch_size", 128)
            units = model_param.get("units", 300)
            units1 = model_param.get("units1", 64)
            units2 = model_param.get("units2", 32)

            for input_type in tqdm.tqdm(INPUT_FEATURES, desc=f"Training: {model_name}", leave=False):
                input_features = config["prometheus_config"].get(input_type, None)
                for interval in tqdm.tqdm(INTERVALS, desc=f"Training:{input_features}", leave=False):
                    model, scaler, normalizer, backup_weights = None, None, None, None

                    input_interval_dir_path = os.path.join(
                        TEST_DIR_PATH,
                        deployment,
                        model_name,
                        input_type,
                        f"{interval}"
                    )
                    data_dir_path = os.path.join(input_interval_dir_path, "data")
                    dir_path = os.path.join(input_interval_dir_path, "model")
                    model_iterations_dir = os.path.join(dir_path, "all_models")

                    os.makedirs(input_interval_dir_path, exist_ok=True)
                    os.makedirs(data_dir_path, exist_ok=True)
                    os.makedirs(dir_path, exist_ok=True)
                    os.makedirs(model_iterations_dir, exist_ok=True)

                    full_model_name = f"{deployment}_{model_name}_{input_type}"
                    model_path = os.path.join(dir_path, "model", f"{full_model_name}_model.h5")

                    log_file_path = os.path.join(input_interval_dir_path, "log.log")
                    init_logs(log_file=log_file_path)

                    interval_dataset = split_dataset_by_interval(service_dataset, interval)

                    for i, data in tqdm.tqdm(
                        enumerate(interval_dataset),
                        total=len(interval_dataset),
                        desc=f"Training: {interval}",
                        leave=False,
                    ):
                        logs("----------")
                        logs(f"Iteration {i + 1}/{len(interval_dataset)} for interval {interval} days...")
                        logs(f"Deployment: {deployment}")

                        input_features = [f for f in input_features if f != "deployment_id"]

                        logs(f"Input Features: {input_features}")
                        logs(f"Target Features: {TARGET}")

                        try:
                            if model is None:
                                if os.path.exists(model_path):
                                    logs("Existing model found on disk -> loading once...")
                                    load_result = load_model(
                                        model_name=full_model_name,
                                        data_path=os.path.join(dir_path, "model")
                                    )
                                    if load_result is not None:
                                        model, scaler, normalizer = load_result
                                        backup_weights = model.get_weights()
                                        logs("Model loaded successfully")
                                    else:
                                        logs("Model exists but could not be loaded -> making new one from scratch")
                                else:
                                    logs("No model found -> making new one from scratch")
                            else:
                                logs("Continuing training on existing model in memory...")

                            X, y, normalize, scaler = preprocess_data_multi(
                                dataframe=data,
                                features=input_features,
                                target_col=TARGET,
                                scaler=scaler,
                                normalize_scaler=normalizer,
                            )
                            X_train, X_test, y_train, y_test = data_split_multi(
                                X=X,
                                y=y,
                                seq_length=seq_length,
                                future_steps=future_steps
                            )

                            if model is None:
                                real_feature_count = X_train.shape[2]
                                logs(f"Building model {full_model_name} with feature_count={real_feature_count}")
                                model = get_model(
                                    model_name=f"{model_name}",
                                    seq_length=seq_length,
                                    feature_count=real_feature_count,
                                    units=units,
                                    units1=units1,
                                    units2=units2
                                )

                            X_train = np.asarray(X_train, dtype=np.float32)
                            X_test = np.asarray(X_test, dtype=np.float32)
                            y_train = np.asarray(y_train, dtype=np.float32)
                            y_test = np.asarray(y_test, dtype=np.float32)

                            train_ds = (
                                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                                .shuffle(min(len(X_train), 100000))
                                .batch(batch_size, drop_remainder=True)
                                .cache()
                                .prefetch(tf.data.AUTOTUNE)
                            )

                            val_ds = (
                                tf.data.Dataset.from_tensor_slices((X_test, y_test))
                                .batch(batch_size, drop_remainder=True)
                                .cache()
                                .prefetch(tf.data.AUTOTUNE)
                            )

                            start_timer = datetime.now()
                            history = model.fit(
                                train_ds,
                                validation_data=val_ds,
                                epochs=EPOCHS,
                                callbacks=get_callbacks(metadata=metadata),
                                batch_size=batch_size,
                                verbose=0
                            )
                            end_timer = datetime.now()

                            # predictions = model.predict(val_ds, verbose=0)
                            predictions = model.predict(X_test, verbose=0, batch_size=batch_size)

                            predictions_rescaled, y_test_rescaled, timestamp = evaluate_model(
                                predictions, y_test, normalize, scaler, dir_path, start_timer, end_timer
                            )

                            save_model(
                                model=model,
                                scaler=scaler,
                                normalizer=normalize,
                                dir_path=model_iterations_dir,
                                model_name=full_model_name
                            )

                            if backup_weights is None:
                                save_model(
                                    model=model,
                                    scaler=scaler,
                                    normalizer=normalize,
                                    dir_path=dir_path,
                                    model_name=full_model_name
                                )
                                backup_weights = model.get_weights()
                            else:
                                model_old = keras.models.clone_model(model)
                                model_old.set_weights(backup_weights)
                                model_old.compile(
                                    loss=Huber(delta=1.0),
                                    optimizer=Adam(learning_rate=3e-5),
                                    metrics=[MeanAbsoluteError(), MeanSquaredError()]
                                )

                                # predictions_old = model_old.predict(val_ds, verbose=0)
                                predictions_old = model_old.predict(X_test, verbose=0, batch_size=batch_size)

                                improvement = compare_models(
                                    y_pred_old=predictions_old,
                                    y_pred_new=predictions,
                                    y_true=y_test,
                                    normalize=normalize,
                                    scaler=scaler
                                )

                                if improvement:
                                    logs("New model is better -> saving new model.")
                                    save_model(
                                        model=model,
                                        scaler=scaler,
                                        normalizer=normalize,
                                        dir_path=dir_path,
                                        model_name=full_model_name
                                    )
                                    backup_weights = model.get_weights()
                                else:
                                    logs("Old model is better or the same -> skip saving new model.")

                                model_old = None

                        except Exception as e:
                            logs("Error inside training loop...")
                            logs(f"Exception: {e}")
                            logs(traceback.format_exc())

def test(test_dataset):
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(BASE, "config.yaml")) as file:
            config = yaml.safe_load(file)

        DEPLOYMENTS = config["app_config"]["deployments"]
        TEST_DIR_PATH = os.path.join(BASE, "offline_tests_out")
        test_log_file = os.path.join(TEST_DIR_PATH, "all_tests_logs.log")
        init_logs(test_log_file)

    except Exception as e:
        logs(f"Could not load config for testing: {e}")
        return

    for deployment in tqdm.tqdm(DEPLOYMENTS, desc="Testing deployments", leave=False):
        service_dataset = test_dataset[test_dataset["deployment"] == deployment].copy()

        if service_dataset.empty:
            logs(f"No test data for deployment={deployment}")
            continue

        for model_name in tqdm.tqdm(MODEL_NAMES, desc=f"Testing: {deployment}", leave=False):
            metadata = config[f"{model_name}_config"]["metadata"]
            model_param = metadata["model_param"]
            
            future_steps = config["prediction_config"]["future_steps"]
            seq_length = model_param.get("seq_length", 128)
            batch_size = model_param.get("batch_size", 128)
            

            for input_type in INPUT_FEATURES:
                for interval in INTERVALS:
                    logs("----------")
                    logs(f"Testing deployment={deployment}, model={model_name}, interval={interval}")

                    input_interval_dir_path = os.path.join(
                        TEST_DIR_PATH,
                        deployment,
                        model_name,
                        input_type,
                        f"{interval}"
                    )

                    model_dir_path = os.path.join(input_interval_dir_path, "model")
                    eval_dir_path = os.path.join(input_interval_dir_path, "test_results")
                    log_file_path = os.path.join(input_interval_dir_path, "log.log")
                    os.makedirs(eval_dir_path, exist_ok=True)

                    full_model_name = f"{deployment}_{model_name}_{input_type}"
                    model_path = os.path.join(model_dir_path, "model", f"{full_model_name}_model.h5")

                    if not os.path.exists(model_path):
                        logs("Model not found -> skipping.")
                        continue

                    try:
                        load_result = load_model(
                            model_name=full_model_name,
                            data_path=os.path.join(model_dir_path, "model")
                        )

                        if load_result is None:
                            logs("Could not load model -> skipping.")
                            continue

                        model, feature_scaler, target_scaler = load_result

                        input_features = config["prometheus_config"].get(input_type, None)
                        input_features = [f for f in input_features if f != "deployment_id"]

                        X, y, target_scaler, feature_scaler = preprocess_data_multi(
                            dataframe=service_dataset,
                            features=input_features,
                            target_col=TARGET,
                            scaler=feature_scaler,
                            normalize_scaler=target_scaler,
                        )

                        if X is None or y is None or len(X) <= seq_length:
                            logs(f"Not enough samples for sequence creation (len={len(X) if X is not None else 'None'})")
                            continue

                        X_test, y_test = create_sequences_only(
                            X=X,
                            y=y,
                            seq_length=seq_length,
                            future_steps=future_steps
                        )

                        start_timer = datetime.now()

                        predictions = model.predict(
                            X_test,
                            verbose=0,
                            batch_size=batch_size
                        )

                        end_timer = datetime.now()

                        predictions_rescaled, y_test_rescaled, timestamp = evaluate_final_model(
                            predictions=predictions,
                            y_test=y_test,
                            target_scaler=target_scaler,
                            dir_path=eval_dir_path,
                            start_timer=start_timer,
                            end_timer=end_timer,
                            log_file_path=log_file_path
                        )

                    except Exception as e:
                        logs(f"Testing failed: {e}")
                        logs(traceback.format_exc())
def main():
    init_console()
    logs("===================================================")
    logs("Starting offline testing...")

    # load synthetic dataset
    train_dataset, test_dataset = load_synthetic_dataset()
        
    # training
    train(train_dataset)
        
    # testing
    test(test_dataset)


    # out end of program
    init_console()
    logs("Offline testing finished...")
    logs("===================================================")
if __name__ == "__main__":
    main()