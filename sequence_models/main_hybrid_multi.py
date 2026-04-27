from contextlib import redirect_stdout
import io
import os
import time
import traceback
# Disable oneDNN optimizations to avoid potential issues with certain operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import yaml
import tensorflow as tf
#  GPU Configuration 
# tf.config.list_physical_devices('GPU')

from datetime import datetime, timedelta
from keras.models import Model # type: ignore
from keras.layers import LSTM, Dense, Activation, Dropout, Conv1D, GRU, Input, concatenate, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from keras.saving import register_keras_serializable

from my_modules.prometheus_connector import load_prometheus_multi
from my_modules.data_util import data_split, data_split_multi
from my_modules.model_utils import compare_models, evaluate_model, get_callbacks, load_model, save_model, wait_for_midnight, wait_some_min
from my_modules.printable_utils import create_graphs, init_logs, logs

@register_keras_serializable()
def swish(x):
    return tf.nn.swish(x)



def build_hybrid_multi_model(seq_length: int, feature_count: int, units: int):
    input_layer = Input(shape=(seq_length, feature_count))

    lstm_branch = LSTM(units)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)

    conv_branch = Conv1D(
        filters=units,
        kernel_size=3,
        padding="same",
        activation="swish"
    )(input_layer)
    conv_branch = GlobalAveragePooling1D()(conv_branch)
    conv_branch = Dropout(0.2)(conv_branch)

    gru_branch = GRU(units)(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)

    concat_layer = concatenate([lstm_branch, conv_branch, gru_branch])

    x = Dense(128, activation="swish")(concat_layer)
    x = Dense(64, activation="swish")(x)
    output_layer = Dense(5)(x)   # linear output for regression

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss=Huber(delta=1.0),
        optimizer=Adam(learning_rate=3e-5),
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )

    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    logs("Model summary:\n" + summary_str)

    return model


# def build_hybrid_multi_model(seq_length: int, feature_count: int, units: int):
#     input_layer = Input(shape=(seq_length, feature_count))

#     lstm_branch = LSTM(units)(input_layer)
#     lstm_branch = Dropout(0.2)(lstm_branch)

#     conv_branch = Conv1D(filters=units, kernel_size=3, padding="same")(input_layer)
#     conv_branch = GlobalAveragePooling1D()(conv_branch)
#     conv_branch = Dropout(0.2)(conv_branch)

#     gru_branch = GRU(units)(input_layer)
#     gru_branch = Dropout(0.2)(gru_branch)

#     concat_layer = concatenate([lstm_branch, conv_branch, gru_branch])
#     output_layer = Dense(128, activation="swish")(concat_layer)
#     output_layer = Dense(64, activation="swish")(output_layer)
#     output_layer = Dense(1)(output_layer)

#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(
#         loss=Huber(delta=1.0),
#         optimizer=Adam(learning_rate=3e-5),
#         metrics=[MeanAbsoluteError(), MeanSquaredError()]
#     )
#     # model.summary()
#     stream = io.StringIO()
#     with redirect_stdout(stream):
#         model.summary()
#     summary_str = stream.getvalue()
#     logs("Model summary:\n" + summary_str)

#     return model


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
        model_name = "hybrid"
        metadata = config[f"{model_name}_config"]["metadata"]
        model_param = metadata["model_param"]

        EPOCHS = int(os.getenv("EPOCHS", model_param.get("epochs", 10)))
        seq_length = model_param.get("seq_length", 128)
        batch_size = model_param.get("batch_size", 128)
        units = model_param.get("units", 300)
        weights = metadata["weights"]

        # Data preprocess paths & params
        directory = os.getenv("SAVE_DIR", config[f"{model_name}_config"]["paths"]["save_dir"])
        dir_path = os.path.join(BASE, directory)
        model_path = os.path.join(dir_path, "model", f"{model_name}_model.h5")
        
        # data for prometheus
        PROMETHEUS_URL = config["prometheus_config"]["url"]
        deployments = config["app_config"]["deployments"]

        # choose input features and target features
        target_name = os.getenv("TARGET_NAME", "single_output")
        if target_name == "single_output":
            input_features = config["prometheus_config"].get("single_feature", None)
            target_col = config["prometheus_config"].get("target_col", None)
            target_features = None
        elif target_name == "multi_input":
            input_features = config["prometheus_config"].get("multi_features", None)
            target_col = config["prometheus_config"].get("target_col", None)
            target_features = None
        elif target_name == "multi_output":
            input_features = config["prometheus_config"].get("multi_features", None)
            target_features = config["prometheus_config"].get("target_features", None)
            target_col = None
        else:
            raise ValueError(f"Unknown target name: {target_name}")

        # timer
        sleep_time = int(os.getenv("SLEEP_TIME", metadata["data_param"]["sleep_time"]))
        # initiate logger
        log_file_path = os.getenv("LOG_FILE", config[f"{model_name}_config"]["paths"]["training_logs"])
        init_logs(log_file=log_file_path)
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # Training
    model, scaler, normalizer, backup_weights, last_model_mtime = None, None, None, None, None
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
                            model, scaler, normalizer = load_result
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
            
            # data, normalize, scaler = load_prometheus(metadata, features, PROMETHEUS_URL, deployments, start_time, end_time, scaler, normalizer)
            # X_train, X_test, y_train, y_test = data_split(data=data, seq_length=seq_length, feature_count=1)
            X, y, normalize, scaler = load_prometheus_multi(
                metadata=metadata,
                features=input_features,
                PROMETHEUS_URL=PROMETHEUS_URL,
                deployments=deployments,
                start_time=start_time,
                end_time=end_time,
                scaler=scaler,
                normalizer=normalizer,
                target=target_col,
                target_features=target_features
            )
            if X is None or y is None or len(X) == 0:
                logs("Empty X/y from connector, skipping.")
                raise RuntimeError("No training data")

            feature_count = X.shape[1]
            X_train, X_test, y_train, y_test = data_split_multi(X, y, seq_length=seq_length)

            if model is None:
                model = build_hybrid_multi_model(seq_length=seq_length, feature_count=feature_count, units=units)
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
            predictions_rescaled, y_test_rescaled, timestamp = evaluate_model(predictions, y_test, normalize, scaler, dir_path, start_timer, end_timer)
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
                    backup_weights = model.get_weights() # Update copy of weights

                else: 
                    logs(f"Old model is better or the same -> skip saving new model.")

                model_old = None
        
        except Exception as e:
            logs(f"Error inside training loop...")
            logs(f"Exception: {e}")
            logs(traceback.format_exc())

        finally:
                wait_for_midnight()

if __name__ == "__main__":
    main()