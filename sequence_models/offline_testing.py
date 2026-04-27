import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import traceback
import numpy as np
import tqdm
import keras
import yaml
import tensorflow as tf

from datetime import datetime, timedelta
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from tensorflow.keras.losses import Huber # type: ignore

from my_modules.prometheus_connector import preprocess_data, preprocess_data_multi
from my_modules.data_util import data_split, data_split_multi
from my_modules.model_utils import compare_models, evaluate_model, get_callbacks, load_model, save_model, wait_for_midnight
from my_modules.printable_utils import create_graphs, init_console, init_logs, logs
from my_modules.offline_testing_utils import load_synthetic_dataset, split_dataset_by_interval

from main_lstm import build_lstm_model
from main_lstm_multi import build_lstm_multi_model
from main_hybrid import build_hybrid_model
from main_hybrid_multi import build_hybrid_multi_model


def get_model(model_name: str, seq_length, feature_count = None, units = None, units1=None, units2=None):
    if model_name == 'lstm_single_feature':         return build_lstm_model(seq_length=seq_length, units1=units1, units2=units2)
    elif model_name == 'lstm_multi_features':        return build_lstm_multi_model(seq_length=seq_length, feature_count=feature_count, units1=units1, units2=units2)
    elif model_name == 'hybrid_single_feature':     return build_hybrid_model(seq_length=seq_length, units=units)
    elif model_name == 'hybrid_multi_features':      return build_hybrid_multi_model(seq_length=seq_length, feature_count=feature_count, units=units)
    else: raise ValueError(f"{model_name} is wrong...")


MODEL_NAMES = ["lstm",'hybrid']
# INPUT_FEATURES = ["single_feature","multi_features"]
# INPUT_FEATURES = ["single_feature"]
INPUT_FEATURES = ["multi_features"]
TARGET = "cpu_percent"
INTERVALS = [
    # 1, # 1 day -> 364 times (every day)
    # 7,  # 7 days -> 52 times (every week)
    31, # 31 days -> 12 times (every month)
    # 364, # 364 days -> 1 time (every year) 
]

def train(train_dataset):
    '''
    main function of incremental learning with LSTM model
    loads old model (if exists)
    incremental learning on old model (if exists, if not -> make new with metada in config.yaml)
    if new model better -> save new model
    all important graphs and data save in folder
    '''
    try:
        # load config.yaml
        # BASE = os.getcwd()
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        DEPLOYMENTS = config["app_config"]["deployments"]
        TEST_DIR_PATH = os.path.join(BASE, "offline_tests_out")

    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # Training
    for model_name in tqdm.tqdm(MODEL_NAMES, desc="Test", leave=False):
        # metadata
        metadata = config[f"{model_name}_config"]["metadata"]
        model_param = metadata["model_param"]

        EPOCHS = int(os.getenv("EPOCHS", model_param.get("epochs", 10)))
        seq_length = model_param.get("seq_length", 128)
        batch_size = model_param.get("batch_size", 128)
        units = model_param.get("units", 300)
        units1 = model_param.get("units1", 64)
        units2 = model_param.get("units2", 32)
        
        weights = metadata["weights"]
        
        # ------------------------------------------------------------------------------------
        for input_type in tqdm.tqdm(INPUT_FEATURES, desc=f"Testing: {model_name}", leave=False):
            for interval in tqdm.tqdm(INTERVALS, desc=f"Testing:{input_type}", leave=False):
                model, scaler, normalizer, backup_weights = None, None, None, None

                input_interval_dir_path = os.path.join(TEST_DIR_PATH, model_name, input_type, f"{interval}")
                data_dir_path = os.path.join(input_interval_dir_path, "data")
                dir_path = os.path.join(input_interval_dir_path, "model")
                model_iterations_dir = os.path.join(dir_path, "all_models")

                os.makedirs(input_interval_dir_path, exist_ok=True)
                os.makedirs(data_dir_path, exist_ok=True)
                os.makedirs(dir_path, exist_ok=True)
                os.makedirs(model_iterations_dir, exist_ok=True)

                full_model_name = f"{model_name}_{input_type}"
                model_path = os.path.join(dir_path, "model", f"{full_model_name}_model.h5")

                # initae logs
                log_file_path = os.path.join(input_interval_dir_path, f"log.log")
                init_logs(log_file=log_file_path)  
                
                interval_dataset = split_dataset_by_interval(train_dataset, interval)
                for i, data in tqdm.tqdm(
                    enumerate(interval_dataset),
                    total=len(interval_dataset), 
                    desc=f"Testing: {interval}", 
                    leave=False,
                ):
                    # start learning
                    logs("----------")
                    logs(f"Iteration {i + 1}/{len(interval_dataset)} for interval {interval} days...")
            
                    input_features = config["prometheus_config"].get(input_type, None)
                    logs(f"Input Features: {input_features}")
                    logs(f"Target Features: {TARGET}")  
                    try:
                        if model is None:
                            if os.path.exists(model_path):
                                logs("Existing model found on disk -> loading once...")
                                load_result = load_model(
                                    model_name=f"{model_name}_{input_type}",
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

                        # --- Preprocess Data
                        

                        if input_type == 'single_feature':
                            data, normalize, scaler = preprocess_data(
                                dataframe=data,
                                features=input_features,
                                deployments=DEPLOYMENTS,
                                target_col=TARGET,
                                scaler=scaler,
                                normalize_scaler=normalizer,
                            )
                            X_train, X_test, y_train, y_test = data_split(data=data, seq_length=seq_length, feature_count=1)
                        else:
                            X, y, normalize, scaler = preprocess_data_multi(
                                dataframe=data,
                                features=input_features,
                                deployments=DEPLOYMENTS,
                                target_col=TARGET,
                                scaler=scaler,
                                normalize_scaler=normalizer,
                            )
                            X_train, X_test, y_train, y_test = data_split_multi(X=X, y=y, seq_length=seq_length)

                        logs(f"input_type={input_type}")
                        logs(f"input_features={input_features}")
                        # logs(f"available prometheus_config keys={list(config['prometheus_config'].keys())}")
                        if model is None:
                            real_feature_count = X_train.shape[2]
                            logs(f"Building model {full_model_name} with feature_count={real_feature_count}")
                            model = get_model(
                                model_name=full_model_name,
                                seq_length=seq_length,
                                feature_count=real_feature_count,
                                units=units,
                                units1=units1,
                                units2=units2
                            )
                        else:
                            logs("Continuing training on existing model...")


                        X_train = np.asarray(X_train, dtype=np.float32)
                        X_test = np.asarray(X_test, dtype=np.float32)
                        y_train = np.asarray(y_train, dtype=np.float32)
                        y_test = np.asarray(y_test, dtype=np.float32)

                        logs(f"X_train shape={X_train.shape}, dtype={X_train.dtype}, MB={X_train.nbytes / 1024 / 1024:.2f}")
                        logs(f"X_test shape={X_test.shape}, dtype={X_test.dtype}, MB={X_test.nbytes / 1024 / 1024:.2f}")
                        logs(f"y_train shape={y_train.shape}, dtype={y_train.dtype}, MB={y_train.nbytes / 1024 / 1024:.2f}")
                        logs(f"y_test shape={y_test.shape}, dtype={y_test.dtype}, MB={y_test.nbytes / 1024 / 1024:.2f}")
                        #  Train the Model
                        train_ds = (
                            tf.data.Dataset.from_tensor_slices((X_train, y_train))
                            .shuffle(min(len(X_train), 100000))
                            .batch(batch_size)
                            .cache()
                            .prefetch(tf.data.AUTOTUNE)
                        )

                        val_ds = (
                            tf.data.Dataset.from_tensor_slices((X_test, y_test))
                            .batch(batch_size)
                            .cache()
                            .prefetch(tf.data.AUTOTUNE)
                        )

                        start_timer = datetime.now()
                        history = model.fit(
                            train_ds, # X_train, y_train,
                            validation_data=val_ds, # validation_data=(X_test, y_test),
                            epochs=EPOCHS, 
                            callbacks=get_callbacks(metadata=metadata),
                            batch_size=batch_size, # batch_size=1024,
                            verbose=0
                        )

                        end_timer = datetime.now()
                        # prediction from model
                        # predictions = model.predict(X_test, verbose=0)
                        predictions = model.predict(val_ds, verbose=0)
                        
                        # evaluate the model 
                        predictions_rescaled, y_test_rescaled, timestamp = evaluate_model(predictions, y_test, normalize, scaler, dir_path, start_timer, end_timer)
                        # create_graphs(
                        #     predictions_rescaled=predictions_rescaled, 
                        #     y_test_rescaled=y_test_rescaled, 
                        #     dir_path=dir_path, 
                        #     timestamp=timestamp, 
                        #     start_pos=0, 
                        #     end_pos=1000
                        # )


                        # ---------------------------------------------------------- save copy
                        save_model(
                            model=model,
                            scaler=scaler,
                            normalizer=normalize,
                            dir_path=model_iterations_dir,
                            model_name=full_model_name
                        )
                        # compere with old model
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
                            # predictions_old = model_old.predict(X_test)
                            predictions_old = model_old.predict(val_ds, verbose=0)

                            improvement = compare_models(
                                    y_pred_old=predictions_old,
                                    y_pred_new=predictions,
                                    y_true=y_test,
                                    normalize=normalize,
                                    scaler=scaler
                                )
                            if improvement:
                                logs(f"New model is better -> saving new model.")
                                save_model(
                                    model=model,
                                    scaler=scaler,
                                    normalizer=normalize,
                                    dir_path=dir_path,
                                    model_name=full_model_name
                                )
                                backup_weights = model.get_weights() # Update copy of weights

                            else: 
                                logs(f"Old model is better or the same -> skip saving new model.")

                            model_old = None
                    
                    except Exception as e:
                        logs(f"Error inside training loop...")
                        logs(f"Exception: {e}")
                        logs(traceback.format_exc())


def test(test_dataset):
    pass

def main():
    init_console()
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
if __name__ == "__main__":
    main()