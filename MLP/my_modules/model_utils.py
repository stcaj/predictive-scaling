import os
import time
from typing import Optional, Tuple
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
from my_modules.printable_utils import logs
from sklearn.metrics import accuracy_score, f1_score, recall_score

def wait_for_midnight() -> None:
    """Sleep until 00:00 of the next day"""
    now = datetime.now()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    wait_seconds = (midnight - now).total_seconds()

    logs(f"Next training at {midnight.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(wait_seconds)
    
def wait_some_min(minutes: int) -> None:
    '''Sleep expected minutes before next prediction / training'''
    now = datetime.now()
    skip = (now + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
    wait_seconds = (skip - now).total_seconds()    
    logs(f"Next prediction at {skip.strftime('%Y-%m-%d %H:%M:%S')}")
    time.sleep(wait_seconds)
    

def load_model(
    model_path: str, 
    scaler_path: str,
    history_path: str
) -> Optional[Tuple[MLPClassifier, StandardScaler, np.ndarray, np.ndarray]]:
    """
    Load an MLPClassifier and StandardScaler from disk.
    
    Parameters:
    - model_path: path to the model 
    - scaler_path: path to the scaler
    - history_path: path to history test data

    Returns
    - clf (model), scaler, test_data, test_labels -> if successfully loaded, or None if any error occurs.
    """
    # Check model file exists
    if not os.path.exists(model_path):
        logs(f"Model file not found at '{model_path}'")
        return None

    # Check scaler file exists
    if not os.path.exists(scaler_path):
        logs(f"Scaler file not found at '{scaler_path}'")
        return None

    # Check test data/lalebs file exists
    test_data, test_labels = None, None
    if history_path != None:
        if not os.path.exists(history_path):
            logs(f"Test history file not found at '{history_path}'")
            return None
        else:
            test_data, test_labels = joblib.load(history_path)
            logs("Test history data loaded successfully.")

    try:
        clf = joblib.load(model_path)
        logs("Model loaded successfully.")

        scaler = joblib.load(scaler_path)
        logs("Scaler loaded successfully.")

        return clf, scaler, test_data, test_labels

    except Exception as e:
        logs(f"Error loading model or scaler: {e}")
        return None

def compare_models(
        model_old, 
        model_new, 
        test_old_data, 
        test_old_labels,
        test_new_data,
        test_new_labels,
        scaler,
        alpha: float = 0.5
    ) -> bool:
    """
    Compare old and new model on old vs new test data.
    Returns True if the new model should be saved.
    
    Parameters:
    - model_old: previous model
    - model_new: updated model
    - test_old_data, test_old_labels: old test set (stability)
    - test_new_data, test_new_labels: new test set (adaptability)
    - scaler: feature scaler
    - alpha: weight for combining old vs new accuracy/F1 (0=only new, 1=only old)
    """

    # Scale
    test_old_data = scaler.transform(test_old_data)
    test_new_data = scaler.transform(test_new_data)

    # Predictions
    y_pred_old_old = model_old.predict(test_old_data)
    y_pred_new_old = model_new.predict(test_old_data)

    y_pred_old_new = model_old.predict(test_new_data)
    y_pred_new_new = model_new.predict(test_new_data)

    # Metrics
    acc_old_old = accuracy_score(test_old_labels, y_pred_old_old)  # old model on old data
    acc_new_old = accuracy_score(test_old_labels, y_pred_new_old)  # new model on old data
    acc_old_new = accuracy_score(test_new_labels, y_pred_old_new)  # old model on new data
    acc_new_new = accuracy_score(test_new_labels, y_pred_new_new)  # new model on new data

    # Weighted score: trade-off between old & new
    score_old = alpha * acc_old_old + (1 - alpha) * acc_old_new
    score_new = alpha * acc_new_old + (1 - alpha) * acc_new_new

    logs("=== SCORE COMPARISON ===")
    print(f"Old Model (trade-off): {score_old:.4f}")
    print(f"New Model (trade-off): {score_new:.4f}")

    # Save decision
    return score_new > score_old

# Multi-step scoring
def _multi_step_score(y_true, y_pred, metric):
    """Computes accuracy or F1 (macro/weighted) across all steps in multi-step predictions."""
    if y_true.ndim == 1:  # single step
        return _score_metric(y_true, y_pred, metric)

    n_steps = min(y_true.shape[1], y_pred.shape[1])
    return np.mean([_score_metric(y_true[:, i], y_pred[:, i], metric) for i in range(n_steps)])

def _score_metric(y_true, y_pred, metric):
    '''Return score by given metric'''
    if metric == "step_accuracy": return accuracy_score(y_true, y_pred)
    elif metric == "f1_macro":      return f1_score(y_true, y_pred, average="macro", zero_division=0)
    elif metric == "f1_weighted":   return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    else: raise ValueError(f"Unsupported metric: {metric}")

# Align number of samples for multi-step outputs
def _align_labels(y_true, y_pred):
    n_samples = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        logs(f"[WARN] Aligning lengths: y_true={len(y_true)}, y_pred={len(y_pred)} -> {n_samples}")
    return y_true[:n_samples], y_pred[:n_samples]

# def compare_multi_models(
#     model_old,
#     model_new,
#     test_old_data,
#     test_old_labels,
#     test_new_data,
#     test_new_labels,
#     scaler_old,
#     scaler_new,
#     alpha: float = 0.5,
#     metric: str = "f1_macro"
# ) -> bool:
#     """
#     Compare old and new model on old vs new test data for incremental multi-step learning.
#     Returns True if the new model should be saved.

#     Parameters
#     ---
#     - model_old: old/best model snapshot
#     - model_new: newly trained candidate model
#     - test_old_data, test_old_labels: frozen old test set for stability
#     - test_new_data, test_new_labels: latest test set for adaptability
#     - scaler_old: scaler paired with model_old
#     - scaler_new: scaler paired with model_new
#     - alpha: trade-off weight (stability vs adaptability)
#     - metric: "accuracy", "f1_macro", "f1_weighted"
#     """

#     # --- Build per-model feature spaces ---
#     X_old_for_old = test_old_data
#     X_new_for_old = test_new_data
#     X_old_for_new = test_old_data
#     X_new_for_new = test_new_data

#     if scaler_old is not None:
#         X_old_for_old = scaler_old.transform(test_old_data)
#         X_new_for_old = scaler_old.transform(test_new_data)

#     if scaler_new is not None:
#         X_old_for_new = scaler_new.transform(test_old_data)
#         X_new_for_new = scaler_new.transform(test_new_data)

#     # --- Predictions ---
#     y_pred_old_old = model_old.predict(X_old_for_old)  # old model on old test (old scaler)
#     y_pred_old_new = model_old.predict(X_new_for_old)  # old model on new test (old scaler)

#     y_pred_new_old = model_new.predict(X_old_for_new)  # new model on old test (new scaler)
#     y_pred_new_new = model_new.predict(X_new_for_new)  # new model on new test (new scaler)

#     # --- Align labels/preds ---
#     y_old, y_pred_old_old = _align_labels(test_old_labels, y_pred_old_old)
#     y_old, y_pred_new_old = _align_labels(test_old_labels, y_pred_new_old)

#     y_new, y_pred_old_new = _align_labels(test_new_labels, y_pred_old_new)
#     y_new, y_pred_new_new = _align_labels(test_new_labels, y_pred_new_new)

#     # --- Metrics ---
#     score_old_old = _multi_step_score(y_old, y_pred_old_old, metric)  # stability old
#     score_new_old = _multi_step_score(y_old, y_pred_new_old, metric)  # stability new
#     score_old_new = _multi_step_score(y_new, y_pred_old_new, metric)  # adaptation old
#     score_new_new = _multi_step_score(y_new, y_pred_new_new, metric)  # adaptation new

#     # --- Trade-off between stability and adaptation ---
#     tradeoff_old = alpha * score_old_old + (1 - alpha) * score_old_new
#     tradeoff_new = alpha * score_new_old + (1 - alpha) * score_new_new

#     logs("=== SCORE COMPARISON ===")
#     logs(f"Metric: {metric}")
#     logs(f"Old Model (trade-off): {tradeoff_old:.4f}")
#     logs(f"New Model (trade-off): {tradeoff_new:.4f}")
#     logs(
#         f"Details: old/old={score_old_old:.4f}, new/old={score_new_old:.4f}, "
#         f"old/new={score_old_new:.4f}, new/new={score_new_new:.4f}"
#     )

#     return tradeoff_new > tradeoff_old





def _scaleup_recall(y_true, y_pred, scaleup_classes):
    """Recall pre scale-up triedy (mean cez kroky)."""
    if y_true.ndim == 1:
        return recall_score(
            y_true, y_pred,
            labels=scaleup_classes,
            average="macro",
            zero_division=0
        )

    n_steps = y_true.shape[1]
    scores = []
    for i in range(n_steps):
        scores.append(
            recall_score(
                y_true[:, i],
                y_pred[:, i],
                labels=scaleup_classes,
                average="macro",
                zero_division=0
            )
        )
    return np.mean(scores)


def _transform_numeric_keep_onehot(X: np.ndarray, scaler, n_deps: int) -> np.ndarray:
    if scaler is None:
        return X
    
    n_numeric = X.shape[1] - n_deps
    X_num = X[:, :n_numeric].astype(float, copy=False)
    X_dep = X[:, n_numeric:].astype(float, copy=False)

    # voliteľná kontrola
    exp = getattr(scaler, "n_features_in_", None)
    if exp is not None and exp != X_num.shape[1]:
        raise ValueError(f"Scaler expects {exp} numeric features, but got {X_num.shape[1]}")

    X_num = scaler.transform(X_num)
    return np.hstack([X_num, X_dep])

def compare_multi_models(
    model_old,
    model_new,
    test_old_data,
    test_old_labels,
    test_new_data,
    test_new_labels,
    scaler_old,
    scaler_new,
    alpha: float = 0.3,         # viac váhy na nové dáta
    eps: float = 0.002,         # tolerancia
    scaleup_classes=(3,4),
    n_deps=10,
):
    """
    Rozhodovanie pre inkrementálne učenie.
    Primárne optimalizuje scale-up recall.
    """

    # ---- škálovanie (numeric-only) ----
    X_old_old = _transform_numeric_keep_onehot(test_old_data, scaler_old, n_deps)
    X_old_new = _transform_numeric_keep_onehot(test_new_data, scaler_old, n_deps)

    X_new_old = _transform_numeric_keep_onehot(test_old_data, scaler_new, n_deps)
    X_new_new = _transform_numeric_keep_onehot(test_new_data, scaler_new, n_deps)
    
    # # ---- škálovanie per model ----
    # if scaler_old is not None:
    #     X_old_old = scaler_old.transform(test_old_data)
    #     X_old_new = scaler_old.transform(test_new_data)
    # else:
    #     X_old_old = test_old_data
    #     X_old_new = test_new_data

    # if scaler_new is not None:
    #     X_new_old = scaler_new.transform(test_old_data)
    #     X_new_new = scaler_new.transform(test_new_data)
    # else:
    #     X_new_old = test_old_data
    #     X_new_new = test_new_data

    # ---- predikcie ----
    y_old_old = model_old.predict(X_old_old)
    y_old_new = model_old.predict(X_old_new)

    y_new_old = model_new.predict(X_new_old)
    y_new_new = model_new.predict(X_new_new)

    # ---- scale-up recall ----
    su_old_old = _scaleup_recall(test_old_labels, y_old_old, scaleup_classes)
    su_new_old = _scaleup_recall(test_old_labels, y_new_old, scaleup_classes)

    su_old_new = _scaleup_recall(test_new_labels, y_old_new, scaleup_classes)
    su_new_new = _scaleup_recall(test_new_labels, y_new_new, scaleup_classes)

    # ---- macro F1 ----
    f1_old_old = _multi_step_score(test_old_labels, y_old_old, "f1_macro")
    f1_new_old = _multi_step_score(test_old_labels, y_new_old, "f1_macro")

    f1_old_new = _multi_step_score(test_new_labels, y_old_new, "f1_macro")
    f1_new_new = _multi_step_score(test_new_labels, y_new_new, "f1_macro")

    # ---- tradeoff ----
    su_trade_old = alpha * su_old_old + (1 - alpha) * su_old_new
    su_trade_new = alpha * su_new_old + (1 - alpha) * su_new_new

    f1_trade_old = alpha * f1_old_old + (1 - alpha) * f1_old_new
    f1_trade_new = alpha * f1_new_old + (1 - alpha) * f1_new_new

    logs("=== SCORE COMPARISON ===")
    logs(f"Scale-up recall OLD: {su_trade_old:.4f}")
    logs(f"Scale-up recall NEW: {su_trade_new:.4f}")
    logs(f"Macro F1 OLD:        {f1_trade_old:.4f}")
    logs(f"Macro F1 NEW:        {f1_trade_new:.4f}")

    # ---- rozhodovanie ----

    # 1) scale-up recall sa nesmie výrazne zhoršiť
    if su_trade_new < su_trade_old - eps:
        logs("REJECT: scale-up recall degraded.")
        return False

    # 2) ak scale-up lepší alebo rovnaký → rozhodni podľa kombinácie
    combined_old = 0.6 * su_trade_old + 0.4 * f1_trade_old
    combined_new = 0.6 * su_trade_new + 0.4 * f1_trade_new

    logs(f"Combined OLD: {combined_old:.4f}")
    logs(f"Combined NEW: {combined_new:.4f}")

    return combined_new > combined_old + eps



















def save_model(
        clf: MLPClassifier,
        model_path: str, 
        scaler: StandardScaler, 
        scaler_path: str
    ) -> None:
    '''
    save model and scaler inside given paths

    Parameters:
    - clf -> model
    - model_path -> dir and file name of model
    - scaler -> scaler
    - scaler_path -> dir and file name of scaler
    '''
    joblib.dump(clf, model_path)
    logs(f"Model saved as '{model_path}'")
    joblib.dump(scaler, scaler_path)
    logs(f"Scaler saved as '{scaler_path}'")


def save_test_data_and_labels(
    test_database,
    test_path: str,
):
    '''
    Save test data and labels and return them when making new ones with old ones.
    Returns new data and labels containing both old and new.
    -> the experiment wont be too long do not need to trim it
    '''
    test_data_new, test_labels_new, test_data_old, test_labels_old = test_database

    # Stack test data vertically (add new rows)
    test_data = np.vstack([test_data_old, test_data_new])

    # Ensure labels are 2D arrays for safe stacking
    test_labels_old = np.atleast_2d(test_labels_old)
    test_labels_new = np.atleast_2d(test_labels_new)

    # Stack labels vertically
    test_labels = np.vstack([test_labels_old, test_labels_new])

    # Save
    joblib.dump((test_data, test_labels), test_path)
    logs("Test data & labels saved as 'history_data.pkl'")

    return test_data, test_labels