import os
import traceback
import yaml
# Disable oneDNN optimizations to avoid potential issues with certain operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
#  GPU Configuration 
tf.config.list_physical_devices('GPU')

from datetime import datetime
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore

from my_modules.data_util import data_split, data_split_old
from my_modules.fastStorage_utils import preprocess_data_from_fastStorage
from my_modules.model_utils import evaluate_model, get_callbacks, save_model
from my_modules.printable_utils import create_graphs, logs

model_name = "lstm"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def build_model(seq_length: int, units1, units2):
    '''
    Build LSTM model

    Parameters:
    - seq_length: length of sequence
    - units1:
    - units2:

    Returns:
    - Sequential model to learn
    '''
    model = Sequential()
    model.add(LSTM(units=units1, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units2, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='swish'))
    model.add(Dense(32, activation='swish'))
    model.add(Dense(1))

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
        units1 = model_param.get("units1", 64)
        units2 = model_param.get("units2", 32)
        
        # metadata only for faststorage
        features = config["fastStorage_config"]["features"]
        
        # Data preprocess paths & params
        input_pattern = os.path.join(os.path.dirname(BASE), config["fastStorage_config"]["fastStorage_patern"]) # should be dir back
        dir_path = os.path.join(BASE, config[f"{model_name}_config"]["paths"]["fastStorage_save_dir"])
        
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)


    # testing faststorage -> not saving model
    model, scaler, normalizer = None, None, None

    # Training
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
    model = build_model(seq_length=seq_length, units1=units1, units2=units2)

    #  Train the Model
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    start_timer = datetime.now()
    history = model.fit(
        train_ds, # X_train, y_train,
        validation_data=val_ds, # validation_data=(X_test, y_test),
        epochs=EPOCHS, 
        callbacks=get_callbacks(metadata=metadata),
        batch_size=batch_size,
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