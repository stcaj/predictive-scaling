import copy
from datetime import datetime, timedelta
import os
import traceback
import numpy as np
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import tqdm
import yaml

from sklearn.neural_network import MLPClassifier
from my_modules.prometheus_connector import load_prometheus
from my_modules.model_utils import compare_multi_models, load_model, save_model, save_test_data_and_labels, wait_some_min
from sklearn.preprocessing import StandardScaler
from my_modules.printable_utils import  init_logs, logs, plot_confusion_matrix_mltiSteps



def MultiOutClass_partial_train(
        clf,
        scaler,
        train_data, 
        train_labels, 
        test_data, 
        test_labels,
        metadata,
        plot_path
):
    """Incremental training of MultiOutput MLPClassifier with partial_fit."""
    # Metadata
    epochs = metadata["model_param"].get("epochs", 10)
    batch_size = metadata["model_param"].get("batch_size", 32)
    hidden_layer_sizes = metadata["model_param"].get("hidden_layer_sizes", (100, 50))
    max_iter = metadata["model_param"].get("max_iter", 200)
    random_state = metadata["model_param"].get("random_state", 42)
    num_classes = len(metadata["data_param"]["class_values"])
    rng = np.random.default_rng(random_state)

    # --- split: numeric part vs one-hot deployment part ---
    n_deps = len(metadata["app_deployments"])  # one-hot dim
    if train_data.shape[1] < n_deps:
        raise ValueError(f"train_data has {train_data.shape[1]} cols but n_deps={n_deps}")

    n_numeric = train_data.shape[1] - n_deps 

    Xtr_num = train_data[:, :n_numeric].astype(float, copy=False)
    Xtr_dep = train_data[:, n_numeric:].astype(float, copy=False)
    Xte_num = test_data[:, :n_numeric].astype(float, copy=False)
    Xte_dep = test_data[:, n_numeric:].astype(float, copy=False)

    # --- scaler len na numeric časti ---
    if scaler is None:
        scaler = StandardScaler()

    scaler.partial_fit(Xtr_num)
    Xtr_num = scaler.transform(Xtr_num)
    Xte_num = scaler.transform(Xte_num)

    # poskladaj späť (one-hot ostáva nezmenený)
    train_data = np.hstack([Xtr_num, Xtr_dep])
    test_data  = np.hstack([Xte_num, Xte_dep])

    # Handle multi-output label shape
    if len(train_labels.shape) == 1:
        train_labels = train_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)

    n_outputs = train_labels.shape[1]
    classes = [np.arange(num_classes) for _ in range(n_outputs)]

    # MLP Classifier
    if clf is None:
        clf = MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
            )
        )

    # Training loop
    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs", leave=False):
        logs(f"Epoch {epoch + 1}/{epochs}...")

        indices = rng.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]

        for start in range(0, len(train_data), batch_size):
            end = start + batch_size
            X_batch = train_data_shuffled[start:end]
            y_batch = train_labels_shuffled[start:end].astype(int)
            clf.partial_fit(X_batch, y_batch, classes=classes)
    logs("Done...")
    # accuracies
    logs("Train accuracy = {:.4f}".format(clf.score(train_data, train_labels)))
    logs("Test accuracy  = {:.4f}".format(clf.score(test_data, test_labels)))
    # Confusion matrix (one per output)
    test_pred = clf.predict(test_data)
    timestamp = datetime.now()

    for step in range(n_outputs):
        class_report = classification_report(test_labels[:, step], test_pred[:, step], digits=4, zero_division=0)
        logs(f"Classification Report for step {step+1}:\n{class_report}")
        
        con_mat = np.zeros((num_classes, num_classes), dtype=int)
        for pred, actual in zip(test_pred[:, step], test_labels[:, step]):
            con_mat[int(actual), int(pred)] += 1

        plot_confusion_matrix_mltiSteps(
            con_mat=con_mat, 
            num_classes=num_classes, 
            path=plot_path, 
            step=step+1,
            timestamp=timestamp, 
            nor=True
        )

    return clf, scaler




# def MultiOutClass_partial_train(
#         clf,
#         scaler,
#         train_data, 
#         train_labels, 
#         test_data, 
#         test_labels,
#         metadata,
#         plot_path
# ):
#     """Incremental training of MultiOutput MLPClassifier with partial_fit."""
#     # Metadata
#     epochs = metadata["model_param"].get("epochs", 10)
#     batch_size = metadata["model_param"].get("batch_size", 32)
#     hidden_layer_sizes = metadata["model_param"].get("hidden_layer_sizes", (100,50))
#     max_iter = metadata["model_param"].get("max_iter", 200)
#     random_state = metadata["model_param"].get("random_state", 42)
#     num_classes = len(metadata["data_param"]["class_values"])
#     rng = np.random.default_rng(random_state)
    
#     # Preprocess
#     if scaler is None:
#         scaler = StandardScaler()

#     scaler.partial_fit(train_data)
#     train_data = scaler.transform(train_data)
#     test_data = scaler.transform(test_data)

#     # Handle multi-output label shape
#     if len(train_labels.shape) == 1:
#         train_labels = train_labels.reshape(-1, 1)
#         test_labels = test_labels.reshape(-1, 1)

#     n_outputs = train_labels.shape[1]
#     classes = [np.arange(num_classes) for _ in range(n_outputs)]

#     # MLP Classifier
#     if clf is None:
#         clf = MultiOutputClassifier(
#             MLPClassifier(
#                 hidden_layer_sizes=hidden_layer_sizes, 
#                 max_iter=max_iter, 
#                 random_state=random_state,
#                 # warm_start=True
#             )
#         )

#     # Training loop
#     for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs", leave=False):
#         logs(f"Epoch {epoch + 1}/{epochs}...")

#         indices = rng.permutation(len(train_data))
#         train_data_shuffled = train_data[indices]
#         train_labels_shuffled = train_labels[indices]

#         for start in range(0, len(train_data), batch_size):
#             end = start + batch_size
#             X_batch = train_data_shuffled[start:end]
#             y_batch = train_labels_shuffled[start:end].astype(int) 
#             clf.partial_fit(X_batch, y_batch, classes=classes)
    
#     logs("Done...")
#     # accuracies
#     logs("Train accuracy = {:.4f}".format(clf.score(train_data, train_labels)))
#     logs("Test accuracy  = {:.4f}".format(clf.score(test_data, test_labels)))
#     # Confusion matrix (one per output)
#     test_pred = clf.predict(test_data)
#     timestamp = datetime.now()

#     for step in range(n_outputs):
#         class_report = classification_report(test_labels[:, step], test_pred[:, step], digits=4, zero_division=0)
#         logs(f"Classification Report for step {step+1}:\n{class_report}")
        
#         con_mat = np.zeros((num_classes, num_classes), dtype=int)
#         for pred, actual in zip(test_pred[:, step], test_labels[:, step]):
#             con_mat[int(actual), int(pred)] += 1

#         plot_confusion_matrix_mltiSteps(
#             con_mat=con_mat, 
#             num_classes=num_classes, 
#             path=plot_path, 
#             step=step+1,
#             timestamp=timestamp, 
#             nor=True
#         )

#     return clf, scaler


def main():
    '''
    main function of incremental learning with one future class prediction
    loads old model (if exists)
    incremental learning on old model (if exists, if not -> make new with metada in config.yaml)
    if new model better -> save new model
    all important graphs and data save in folder
    '''

    # load metadata/paths to dirs
    try:
        # load config.yaml
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        # metadata
        metadata = config["metadata"]
        paths = config["paths"]
        model_name = "mlp_multiStep"

        # model locations
        model_dir = os.path.join(BASE, paths[model_name]["model_path"])
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}_model.pkl") # path to saved model 
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl") # path to saved scaler
        history_path = os.path.join(model_dir, "history_data.pkl")
        
        # timer
        sleep_time = int(os.getenv("SLEEP_TIME", metadata["model_param"]["training_sleep_minutes"]))
        first_sleep = int(os.getenv("FIRST_SLEEP", 0))
        # initiate logger
        log_file_path = os.getenv("LOG_FILE", paths["logs"]["train_log_path"])
        init_logs(log_file=log_file_path)

        # initate logger
    except Exception as e:
        print(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)


    # load and test saved model 
    clf, clf_copy, scaler, test_data_old, test_labels_old, last_model_mtime = None, None, None, None, None, None
    
    if first_sleep > 0:
        wait_some_min(first_sleep)

    while True:
        logs("----------")
        try:
            if os.path.exists(model_path):
                current_mtime = os.path.getmtime(model_path)
                if last_model_mtime is None or current_mtime > last_model_mtime:
                    logs("Detected new model file, reloading...")
                    try:
                        load_result = load_model(model_path=model_path,
                                                scaler_path=scaler_path,
                                                history_path=history_path)
                        if load_result is not None:
                            clf, scaler, test_data_old, test_labels_old = load_result
                            clf_copy = copy.deepcopy(clf)
                            logs("Model reloaded successfully")
                        last_model_mtime = current_mtime
                    except Exception as e:
                        logs(f"Error reloading model: {e}")
            else:
                logs("No model found -> making new one from scratch")


            # load data 
            end_time = datetime.now().replace(second=0, microsecond=0)
            start_time = end_time - timedelta(minutes=sleep_time)        
            train_data, train_labels, test_data, test_labels = load_prometheus(
                metadata=metadata,
                model_name=model_name,
                start_time=start_time,
                end_time=end_time,
                # smote=True
            )

            if len(train_data) > 0:
                # train model 
                data_dir_path = os.path.join(BASE, paths[model_name]["data_path"]) 
                clf, scaler = MultiOutClass_partial_train(
                                    clf=clf,
                                    scaler=scaler,
                                    train_data=train_data, 
                                    train_labels=train_labels, 
                                    test_data=test_data, 
                                    test_labels=test_labels,
                                    metadata=metadata,
                                    plot_path=data_dir_path
                                )


                # save new model & scaler 
                if clf_copy is None:
                    logs("Old model does not exits -> saving new model.")
                    test_data_old, test_labels_old = test_data, test_labels
                    save_model(
                        clf=clf, 
                        model_path=model_path, 
                        scaler=scaler, 
                        scaler_path=scaler_path
                    )
                    clf_copy = copy.deepcopy(clf)
                    test_data_old, test_labels_old = save_test_data_and_labels(test_database=[test_data, test_labels, test_data_old, test_labels_old], test_path=history_path)

                else:
                    improvement = compare_multi_models(
                                    model_old=clf_copy, 
                                    model_new=clf,
                                    test_old_data= test_data_old,
                                    test_old_labels= test_labels_old,
                                    test_new_data=test_data, 
                                    test_new_labels=test_labels, 
                                    scaler=scaler
                                )
                    if improvement:
                        logs("New model is better -> saving new model.")
                        save_model(
                            clf=clf, 
                            model_path=model_path, 
                            scaler=scaler, 
                            scaler_path=scaler_path
                        )
                        clf_copy = copy.deepcopy(clf)
                        test_data_old, test_labels_old = save_test_data_and_labels(test_database=[test_data, test_labels, test_data_old, test_labels_old], test_path=history_path)

                    else:
                        logs("Old model is better or the same -> skip saving new model.")
                        # Restore clf to the best model
                        clf = copy.deepcopy(clf_copy)
            else:
                logs("Not enough data...")
        
        except Exception as e:
            logs(f"Error inside training loop...")
            logs(f"Exception: {e}")
            logs(traceback.format_exc())

        finally:    
            wait_some_min(sleep_time)



# start main
if __name__ == '__main__':
    main()
