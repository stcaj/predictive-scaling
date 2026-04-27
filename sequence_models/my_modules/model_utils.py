import os
import csv
import time
from typing import Tuple
import joblib
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.metrics import MeanSquaredError, MeanAbsoluteError # type: ignore
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from my_modules.printable_utils import logs
# from my_modules.offline_testing_utils import compute_epoch_durations

def wait_some_min(minutes: int) -> None:
    '''Sleep expected minutes before next prediction / training'''
    now = datetime.now()
    skip = (now + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
    wait_seconds = (skip - now).total_seconds()    
    logs(f"Next prediction / training at {skip.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(wait_seconds)

def wait_for_midnight() -> None:
    """Sleep until 00:00 of the next day"""
    now = datetime.now()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    wait_seconds = (midnight - now).total_seconds()

    logs(f"Next training at {midnight.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(wait_seconds)

class LogCallback(Callback):
    def on_train_begin(self, logs_dict=None):
        logs("Training started")

    def on_epoch_end(self, epoch, logs_dict=None):
        if logs_dict is not None:
            metrics = ", ".join([f"{k}={v}" for k, v in logs_dict.items()])
            logs(f"Epoch {epoch+1}: {metrics}")

    def on_train_end(self, logs_dict=None):
        logs("Training finished")


def get_callbacks(metadata) -> Tuple:
    '''
    Returns Early Stopping, Reduce LR on Plateau, and Logging callback
    '''
    early_stop_param = metadata["EarlyStopping_param"]
    lr_red_param = metadata["ReduceLROnPlateau_param"]
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=early_stop_param["monitor"], 
        patience=early_stop_param["patience"], 
        restore_best_weights=early_stop_param["restore_best_weights"], 
        # verbose=early_stop_param["verbose"], 
        verbose=0, 
        mode=early_stop_param["mode"]
    )
    
    lr_red = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=lr_red_param["monitor"], 
        factor=lr_red_param["factor"], 
        patience=lr_red_param["patience"], 
        # verbose=lr_red_param["verbose"], 
        verbose=0,
        min_lr=float(lr_red_param["min_lr"]) 
    )

    log_callback = LogCallback()

    return early_stopping, lr_red, log_callback

def evaluate_model(
        predictions: np.ndarray,
        y_test: np.ndarray,
        normalize,
        scaler,
        dir_path: str,
        start_timer: datetime,
        end_timer: datetime
    ) -> tuple:
    """
    Evaluate regression model for single-step or multi-step output.
    normalize = target scaler
    scaler = feature scaler (unused here for y inverse transform)
    """

    pred_shape = predictions.shape
    true_shape = y_test.shape

    # flatten only for inverse_transform of 1-feature scaler
    predictions_2d = predictions.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    try:
        predictions_rescaled = normalize.inverse_transform(predictions_2d).reshape(pred_shape)
        y_test_rescaled = normalize.inverse_transform(y_test_2d).reshape(true_shape)
        logs("Inverse transform applied correctly (target scaler only).")
    except Exception as e:
        logs(f"Target inverse transform failed ({e}); using raw values.")
        predictions_rescaled = predictions
        y_test_rescaled = y_test

    errors = predictions_rescaled - y_test_rescaled
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled.reshape(-1), predictions_rescaled.reshape(-1))
    max_error = np.max(errors)
    min_error = np.min(errors)

    logs("=== METRICS ===")
    for step in range(predictions_rescaled.shape[1]):
        step_mae = np.mean(np.abs(predictions_rescaled[:, step] - y_test_rescaled[:, step]))
        logs(f"MAE step {step+1}: {step_mae:.6f}")
    logs(f"MAE: {mae:.6f}")
    logs(f"MSE: {mse:.6f}")
    logs(f"RMSE: {rmse:.6f}")
    logs(f"R²: {r2:.6f}")
    logs(f"Max Err: {max_error:.6f}")
    logs(f"Min Err: {min_error:.6f}")
    logs(f"y_true range: {y_test_rescaled.min():.4f} - {y_test_rescaled.max():.4f}")
    logs(f"y_pred range: {predictions_rescaled.min():.4f} - {predictions_rescaled.max():.4f}")

    results_path = os.path.join(dir_path, "data", "results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    file_exists = os.path.isfile(results_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(results_path, mode="a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow([
                "Timestamp", "MAE", "MSE", "RMSE", "R2", "Max", "Min",
                "Start time", "End time", "Training time (s)"
            ])
        writer.writerow([
            timestamp, f"{mae:.8f}", f"{mse:.8f}", f"{rmse:.8f}", f"{r2:.8f}",
            f"{max_error:.8f}", f"{min_error:.8f}",
            start_timer.strftime("%Y-%m-%d %H:%M:%S"),
            end_timer.strftime("%Y-%m-%d %H:%M:%S"),
            (end_timer - start_timer).total_seconds()
        ])

    return predictions_rescaled, y_test_rescaled, timestamp


def evaluate_final_model(
        predictions: np.ndarray,
        y_test: np.ndarray,
        target_scaler,
        dir_path: str,
        start_timer: datetime,
        end_timer: datetime,
        log_file_path: str,
    ) -> tuple:

    pred_shape = predictions.shape
    true_shape = y_test.shape

    predictions_2d = predictions.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    try:
        predictions_rescaled = target_scaler.inverse_transform(predictions_2d).reshape(pred_shape)
        y_test_rescaled = target_scaler.inverse_transform(y_test_2d).reshape(true_shape)
        logs("Inverse transform applied correctly (target_scaler only).")
    except Exception as e:
        logs(f"Target inverse transform failed ({e}); using raw values.")
        predictions_rescaled = predictions
        y_test_rescaled = y_test

    errors = predictions_rescaled - y_test_rescaled
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled.reshape(-1), predictions_rescaled.reshape(-1))
    max_error = np.max(errors)
    min_error = np.min(errors)

    epoch_count = None
    sec_avg = None
    total_sec = None

# --------------------------------------------------------------- uncomment for learning time
    # try:
    #     with open(log_file_path, "r", encoding="utf-8") as f:
    #         res = compute_epoch_durations(f.readlines())
    #         epoch_count = res["count"]
    #         sec_avg = res["avg_sec"]
    #         total_sec = res["total_sec"]
    # except Exception as e:
    #     logs(f"Could not open/parse log.log file: {e}")

    logs("=== METRICS ===")
    for step in range(predictions_rescaled.shape[1]):
        step_mae = np.mean(np.abs(predictions_rescaled[:, step] - y_test_rescaled[:, step]))
        logs(f"MAE step {step+1}: {step_mae:.6f}")
    logs(f"MAE: {mae:.6f}")
    logs(f"MSE: {mse:.6f}")
    logs(f"RMSE: {rmse:.6f}")
    logs(f"R²: {r2:.6f}")
    logs(f"Max Err: {max_error:.6f}")
    logs(f"Min Err: {min_error:.6f}")
    logs(f"y_true range: {y_test_rescaled.min():.4f} - {y_test_rescaled.max():.4f}")
    logs(f"y_pred range: {predictions_rescaled.min():.4f} - {predictions_rescaled.max():.4f}")

    if epoch_count is not None:
        logs(f"Epoch count: {epoch_count}")
        logs(f"Sec avg: {sec_avg}")
        logs(f"Total seconds: {total_sec}")

    results_path = os.path.join(dir_path, "results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    file_exists = os.path.isfile(results_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(results_path, mode="a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow([
                "Timestamp", "MAE", "MSE", "RMSE", "R2", "Max", "Min",
                "Epochs", "Sec AVG", "Total sec"
            ])
        writer.writerow([
            timestamp, f"{mae:.8f}", f"{mse:.8f}", f"{rmse:.8f}", f"{r2:.8f}",
            f"{max_error:.8f}", f"{min_error:.8f}",
            epoch_count,
            sec_avg,
            total_sec
        ])

    return predictions_rescaled, y_test_rescaled, timestamp

# Optional x100 scale correction if units differ
def maybe_fix_units(y_t, y_p):
    eps = 1e-8
    mt, mp = np.max(np.abs(y_t)) + eps, np.max(np.abs(y_p)) + eps
    ratio = max(mt, mp) / max(min(mt, mp), eps)
    if 50 <= ratio <= 200:
        if mp > mt:
            y_p /= 100.0
            logs("Unit auto-fix: divided predictions by 100.")
        else:
            y_p *= 100.0
            logs("Unit auto-fix: multiplied predictions by 100.")
    return y_t, y_p

def compare_models(
        y_pred_old: np.ndarray,
        y_pred_new: np.ndarray,
        y_true: np.ndarray,
        normalize=None,
        scaler=None,
        min_rel_improvement = 0.005
    ) -> bool:
    """
    Compare old vs new model using RMSE improvement.
    Returns True if the new model is sufficiently better.
    """

    # flatten arrays
    y_true = y_true.flatten()
    y_pred_old = y_pred_old.flatten()
    y_pred_new = y_pred_new.flatten()

    # inverse transforms
    def _inverse(arr):
        arr = arr.reshape(-1,1)
        if scaler is not None:
            arr = scaler.inverse_transform(arr)
        if normalize is not None:
            arr = normalize.inverse_transform(arr)
        return arr.flatten()

    try:
        y_true = _inverse(y_true)
        y_pred_old = _inverse(y_pred_old)
        y_pred_new = _inverse(y_pred_new)
        logs("Inverse transform applied in compare_models.")
    except Exception as e:
        logs(f"Inverse transform failed: {e}")

    # compute RMSE
    rmse_old = np.sqrt(mean_squared_error(y_true, y_pred_old))
    rmse_new = np.sqrt(mean_squared_error(y_true, y_pred_new))

    # relative improvement
    rel_improvement = (rmse_old - rmse_new) / max(rmse_old, 1e-8)

    logs("=== RMSE Comparison ===")
    logs(f"RMSE old: {rmse_old:.6f}")
    logs(f"RMSE new: {rmse_new:.6f}")
    logs(f"Relative improvement: {rel_improvement:.4%}")

    if rel_improvement > min_rel_improvement:
        logs("New model is significantly better.")
        return True
    else:
        logs("Improvement too small -> keeping old model.")
        return False

# save model
def save_model(
        model, 
        scaler,
        normalizer, 
        dir_path: str, 
        model_name: str
    ) -> None:
    '''
    save model, normalizer & scaler

    Parameters:
    - model: model to save 
    - scaler: scaler to save
    - normalizer: normalizer to save
    - dir_path: path to directory for saving data
    - model_name: name of the model used
    '''
    
    os.makedirs(os.path.join(dir_path, "model"), exist_ok=True)
    logs(f"Saving model inside '{dir_path}'")

    model.save(os.path.join(dir_path, "model", f"{model_name}_model.h5"))
    # model.save(os.path.join(dir_path, "model", f"{model_name}_model.h5"), custom_objects={'swish': swish})
    logs(f"Model saved as '{model_name}_model.h5'")
    joblib.dump(scaler, os.path.join(dir_path, "model", f"{model_name}_scaler.pkl"))
    logs(f"Scaler saved as '{model_name}_scaler.pkl'")
    joblib.dump(normalizer, os.path.join(dir_path, "model", f"{model_name}_normalizer.pkl"))
    logs(f"Scaler saved as '{model_name}_normalizer.pkl'")

def load_model(model_name: str, data_path: str):
    """
    Load a saved Keras model, scaler, and normalizer.
    Returns (model, scaler, normalizer) or None if any error occurs.
    """
    model_path = os.path.join(data_path, f"{model_name}_model.h5")
    normalizer_path = os.path.join(data_path, f"{model_name}_normalizer.pkl")
    scaler_path = os.path.join(data_path, f"{model_name}_scaler.pkl")

    # Check paths
    if not os.path.exists(model_path):
        logs(f"Model path not found: '{model_path}'")
        return None
    if not os.path.exists(normalizer_path) or not os.path.exists(scaler_path):
        logs("Normalizer or scaler file not found")
        return None

    try:
        model = keras_load_model(model_path, compile=False)
        model.compile(
            loss=Huber(delta=1.0),
            optimizer=Adam(learning_rate=3e-5),
            metrics=[MeanAbsoluteError(), MeanSquaredError()]
        )
        logs("Model loaded and compiled successfully.")
    except Exception as e:
        logs(f"Error loading model: {e}")
        return None

    try:
        scaler = joblib.load(scaler_path)
        normalizer = joblib.load(normalizer_path)
        logs("Scaler and normalizer loaded successfully.")
    except Exception as e:
        logs(f"Error loading scaler/normalizer: {e}")
        return None

    return model, scaler, normalizer