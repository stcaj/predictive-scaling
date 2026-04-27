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
from keras.layers import LSTM, Dense, Activation, Dropout, Conv1D, GRU, Input, concatenate, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from tensorflow.keras.losses import Huber # type: ignore

from my_modules.data_util import data_split, data_split_old
from my_modules.fastStorage_utils import preprocess_data_from_fastStorage
from my_modules.model_utils import evaluate_model, get_callbacks
from my_modules.printable_utils import create_graphs, logs
from my_modules.model_utils import save_model

model_name = "hybrid"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

# Build Hybrid Model
def build_model(seq_length: int, units):
  
    input_layer = Input(shape=(seq_length, 1))

    lstm_branch = LSTM(units)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)

    conv_branch = Conv1D(filters=units, kernel_size=3)(input_layer)
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
        loss=Huber(delta=1.0),#   loss = 'mae', 
        optimizer = Adam(learning_rate=0.00003),
        metrics=[MeanSquaredError(), MeanAbsoluteError()]
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
        features = config["fastStorage_config"]["features"]

        seq_length = model_param.get("seq_length", 128)
        batch_size = model_param.get("batch_size", 128)
        SAMPLE_SIZE_AVERAGE = model_param.get("sample_size_averege", 30)
        units = model_param.get("units", 300)

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
        data, normalize, scaler = preprocess_data_from_fastStorage(input_pattern, features, SAMPLE_SIZE_AVERAGE, scaler, normalizer)

    except Exception as e:
        logs("Could not load data")
        logs(f"Exception: {e}")
        logs(traceback.format_exc())
        exit(-1)

    if len(data) == 0:
        raise ValueError("No data found. Please check the input pattern and ensure that the CSV files are present.")
        
    X_train, X_test, y_train, y_test = data_split(data=data, seq_length=seq_length, feature_count=1)

    # No previous model -> create a fresh model
    model = build_model(seq_length=seq_length, units=units)

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
        start_pos=0, 
        end_pos=1000
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