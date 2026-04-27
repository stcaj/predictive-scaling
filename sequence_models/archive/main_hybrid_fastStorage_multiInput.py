import os
import traceback
# Disable oneDNN optimizations to avoid potential issues with certain operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yaml
import tensorflow as tf
#  GPU Configuration 
# tf.config.list_physical_devices('GPU')

from datetime import datetime
from tensorflow.python.keras.activations import swish
from keras.models import Model # type: ignore
from keras.layers import LSTM, Dense, Dropout, Conv1D, GRU, Input, concatenate, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from tensorflow.keras.losses import Huber # type: ignore

from my_modules.data_util import data_split, data_split_multi
from my_modules.fastStorage_utils import preprocess_data_from_fastStorage_multi_input
from my_modules.model_utils import evaluate_model, get_callbacks, save_model
from my_modules.printable_utils import create_graphs, logs

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
model_name = "hybrid"

def build_model(seq_length: int, feature_count: int, units: int):
    input_layer = Input(shape=(seq_length, feature_count))

    lstm_branch = LSTM(units)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)

    conv_branch = Conv1D(filters=units, kernel_size=3, padding="same")(input_layer)
    conv_branch = GlobalAveragePooling1D()(conv_branch)
    conv_branch = Dropout(0.2)(conv_branch)

    gru_branch = GRU(units)(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)

    concat_layer = concatenate([lstm_branch, conv_branch, gru_branch])
    output_layer = Dense(128, activation='swish')(concat_layer)
    output_layer = Dense(64, activation='swish')(output_layer)
    output_layer = Dense(1, activation='linear')(output_layer)
    # output_layer = Activation(swish)(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss=Huber(delta=1.0),
        optimizer=Adam(learning_rate=3e-5),
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )
    model.summary()
    return model


def main():
    try:
        EPOCHS = 10
        # load config.yaml
        # BASE = os.getcwd()
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, CONFIG_PATH)) as file:
            config = yaml.safe_load(file)
            
        # metadata
        metadata = config[f"{model_name}_config"]["metadata"]
        model_param = metadata["model_param"]

        seq_length = model_param.get("seq_length", 128)
        batch_size = model_param.get("batch_size", 128)
        SAMPLE_SIZE_AVERAGE = model_param.get("sample_size_averege", 30)
        units = model_param.get("units", 300)

        # metadata only for faststorage
        features = config["fastStorage_config"]["multi_features"]
        target_features = config["fastStorage_config"]["target_col"]
        graph_name = "hybrid_multiInput"

        # Data preprocess paths & params
        input_pattern = os.path.join(os.path.dirname(BASE), config["fastStorage_config"]["fastStorage_patern"]) # should be dir back
        dir_path = os.path.join(BASE, config[f"{model_name}_config"]["paths"]["fastStorage_save_dir"])
        
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # only testing fastStorage -> not saving model
    model, scaler, normalizer = None, None, None

    # Train model
    try:
        # --- Preprocess Data -> FastStorage
        X, y, normalize, scaler = preprocess_data_from_fastStorage_multi_input(
                                    input_pattern=input_pattern, 
                                    input_features=features,
                                    target_features=target_features, 
                                    SAMPLE_SIZE_AVERAGE=SAMPLE_SIZE_AVERAGE, 
                                    scaler=scaler, 
                                    normalize_scaler=normalizer
                                )
        
    except Exception as e:
        logs("Could not load data")
        logs(f"Exception: {e}")
        logs(traceback.format_exc())
        exit(-1)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data found. Please check the input pattern and ensure that the CSV files are present.")
        
    X_train, X_test, y_train, y_test = data_split_multi(X=X, y=y, seq_length=seq_length)

    # No previous model -> create a fresh model
    model = build_model(seq_length=seq_length, feature_count=X.shape[1], units=units)

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
        verbose=1
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
        start_pos=1300, 
        end_pos=2000,
        name=graph_name
    )
        
    save_model(
        model=model,
        scaler=scaler,
        normalizer=normalize,
        dir_path=dir_path,
        model_name=model_name + f"_{timestamp}"
    )           

if __name__ == "__main__":
    main()