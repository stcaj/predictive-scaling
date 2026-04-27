from contextlib import redirect_stdout
import io
import os
import traceback
import yaml

import keras
# Disable oneDNN optimizations to avoid potential issues with certain operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
#  GPU Configuration 
tf.config.list_physical_devices('GPU')

from datetime import datetime, timedelta
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from keras.saving import register_keras_serializable

from my_modules.data_util import data_split
from my_modules.model_utils import compare_models, evaluate_model, get_callbacks, load_model, save_model, wait_for_midnight
from my_modules.printable_utils import create_graphs, init_logs, logs
from my_modules.prometheus_connector import load_prometheus


@register_keras_serializable()
def swish(x):
    return tf.nn.swish(x)

def build_lstm_model(seq_length: int, units1, units2):
    '''Build LSTM model'''
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(units1, return_sequences=True),
        Dropout(0.2),
        LSTM(units2),
        Dropout(0.2),
        Dense(64, activation='swish'),
        Dense(32, activation='swish'),
        Dense(1)
    ])

    model.compile(
        loss=Huber(delta=1.0),
        optimizer=Adam(learning_rate=3e-5),
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )
    # model.summary()
    # capture summary as string
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    logs("Model summary:\n" + summary_str, console=False)

    return model


def main():
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
            
        # metadata
        model_name = "lstm"
        metadata = config[f"{model_name}_config"]["metadata"]
        model_param = metadata["model_param"]

        EPOCHS = int(os.getenv("EPOCHS", model_param.get("epochs", 10)))
        seq_length = model_param.get("seq_length", 128)
        batch_size = model_param.get("batch_size", 128)
        units1 = model_param.get("units1", 64)
        units2 = model_param.get("units2", 32)
        weights = metadata["weights"]
        
        # data only for prometheus
        PROMETHEUS_URL = config["prometheus_config"]["url"]
        deployments = config["app_config"]["deployments"]

        # choose input features and target features
        input_features = config["prometheus_config"].get("single_feature", None)
        target_col = config["prometheus_config"].get("target_col", None)
        target_features = None

    
        # Data preprocess paths & params
        directory = os.getenv("SAVE_DIR", config[f"{model_name}_config"]["paths"]["save_dir"])
        dir_path = os.path.join(BASE, directory)
        model_path = os.path.join(dir_path, "model", f"{model_name}_model.h5")
        # timer
        sleep_time = int(os.getenv("SLEEP_TIME", metadata["data_param"]["sleep_time"]))
        # initiate logger
        log_file_path = os.getenv("LOG_FILE", config[f"{model_name}_config"]["paths"]["training_logs"])
        init_logs(log_file=log_file_path)
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)


    # Training
    model, scaler, normalize, backup_weights, last_model_mtime = None, None, None, None, None
    while True:
        logs("----------")
        logs(f"Input Features: {input_features}")
        if target_col is not None:
            logs(f"Target Features: {target_col}")
        if target_features is not None:
            logs(f"Target Features: {target_features}")
        try:
            if os.path.exists(model_path):
                current_mtime = os.path.getmtime(model_path)
                if last_model_mtime is None or current_mtime > last_model_mtime:
                    logs("Detected new model file, reloading...")
                    try:
                        load_result = load_model(model_name=model_name, data_path=os.path.join(dir_path, "model"))
                        if load_result is not None:
                            model, scaler, normalize = load_result
                            backup_weights = model.get_weights()
                            logs("Model reloaded successfully")
                        last_model_mtime = current_mtime
                    except Exception as e:
                        logs(f"Error reloading model: {e}")
            else:
                logs("No model found -> making new one from scratch")


            # --- Preprocess Data -> Prometheus
            end_time = datetime.now().replace(second=0, microsecond=0)
            start_time = end_time - timedelta(minutes=sleep_time)
            data, normalize, scaler = load_prometheus(
                                        metadata=metadata, 
                                        features=input_features, 
                                        PROMETHEUS_URL=PROMETHEUS_URL, 
                                        deployments=deployments, 
                                        start_time=start_time, 
                                        end_time=end_time, 
                                        scaler=scaler, 
                                        normalizer=normalize
                                    )
            
            if len(data) > 0:        
                X_train, X_test, y_train, y_test = data_split(data=data, seq_length=seq_length, feature_count=1)
                if model is None: 
                    model = build_lstm_model(seq_length=seq_length, units1=units1, units2=units2)
                else: 
                    logs("Continuing training on existing model...")
    
                #  Train the Model
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
                predictions = model.predict(X_test)

                # evaluate the model 
                predictions_rescaled, y_test_rescaled, timestamp = evaluate_model(
                                                                        predictions=predictions, 
                                                                        y_test=y_test, 
                                                                        normalize=normalize, 
                                                                        scaler=scaler, 
                                                                        dir_path=dir_path, 
                                                                        start_timer=start_timer, 
                                                                        end_timer=end_timer
                                                                    )
                create_graphs(
                    predictions_rescaled=predictions_rescaled, 
                    y_test_rescaled=y_test_rescaled, 
                    dir_path=dir_path, 
                    timestamp=timestamp, 
                    start_pos=0, 
                    end_pos=1000
                )


                # compere with old model
                if backup_weights is None:
                    save_model(
                        model=model,
                        scaler=scaler,
                        normalizer=normalize,
                        dir_path=dir_path,
                        model_name=model_name
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
                    predictions_old = model_old.predict(X_test)

                    improvement = compare_models(
                                    y_pred_old=predictions_old, 
                                    y_pred_new=predictions, 
                                    y_true=y_test, 
                                    weights=weights,
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
                            model_name=model_name
                        )
                        # Update copy of weights
                        backup_weights = model.get_weights()

                    else: 
                        logs(f"Old model is better or the same -> skip saving new model.")

                    model_old = None
            else:
                logs("Process fot empty data...")

        except Exception as e:
            logs(f"Error inside training loop...")
            logs(f"Exception: {e}")
            logs(traceback.format_exc())

        finally:    
                wait_for_midnight()

if __name__ == "__main__":
    main()