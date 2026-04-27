import copy
from datetime import datetime, timedelta
import os
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error, root_mean_squared_error
import yaml
import time

from sklearn.neural_network import MLPClassifier
from my_modules.prometheus_connector import load_prometheus
from my_modules.model_utils import compare_models, load_model, save_model, save_test_data_and_labels
from sklearn.preprocessing import StandardScaler
from my_modules.printable_utils import init_logs, logs, plot_confusion_matrix


def MLPClass_partial_train(
    clf: Optional[MLPClassifier],
    scaler: Optional[StandardScaler],
    kuber_data,
    metadata: dict,
    path: str
) -> Tuple[MLPClassifier, StandardScaler]:
    """
    Incremental training of an MLPClassifier with partial_fit.
    Returns updated classifier and scaler.

    Parameters:
    - clf: MLPClassifier
    - scaler: Standart Scaler
    - kuber_data = train_data, train_labels, test_data, test_labels
    - metadata: metadata saved inside config.yaml
    - path: path to for plot dir

    Returns:
    - clf, scaler: new learned MLPClassifier and Scaler
    """
    # extract data
    train_data, train_labels, test_data, test_labels = kuber_data
    
    # Extract metadata
    epochs = metadata["model_param"].get("epochs", 10)
    batch_size = metadata["model_param"].get("batch_size", 32)
    hidden_layer_sizes = metadata["model_param"].get("hidden_layer_sizes", (100,50))
    random_state = metadata["model_param"].get("random_state", 42)
    max_iter = metadata["model_param"].get("max_iter", 200)

    # initate logger
    init_logs(log_file="MLP/logs/mlp_model_training_fastStorage_oneStep.log")
    
    # num_classes = len(np.unique(train_labels))
    num_classes = len(metadata["fastStorage_param"]["class_values"])
    logs(f"Starting partial_fit training with [{num_classes}] classes...")


    # fixed class set from config.yaml
    classes = np.arange(num_classes)

    # initialize scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
    
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # initialize classifier if not provided
    if clf is None:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter
        )

    # training loop
    for epoch in range(epochs):
        logs(f"Epoch {epoch + 1}/{epochs}...")

        # shuffle data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]

        # train in mini-batches
        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            X_batch = train_data_shuffled[start:end]
            y_batch = train_labels_shuffled[start:end]

            if epoch == 0 and start == 0:
                clf.partial_fit(X_batch, y_batch, classes=classes)
            else:
                clf.partial_fit(X_batch, y_batch)

    logs("Done...")
    logs("Train accuracy = {:.4f}".format(clf.score(train_data, train_labels)))
    logs("Test accuracy  = {:.4f}".format(clf.score(test_data, test_labels)))

    # Confusion matrix and classification report
    test_pred = clf.predict(test_data)

    # Classification report
    logs("Classification Report:")
    logs(classification_report(test_labels, test_pred, digits=4))

    # Confusion matrix
    con_mat = np.zeros((num_classes, num_classes))
    for pred, actual in zip(test_pred, test_labels):
        con_mat[int(actual)][int(pred)] += 1

    # RMSE and MAE
    rmse = root_mean_squared_error(test_labels, test_pred)
    mae = mean_absolute_error(test_labels, test_pred)
    logs("RMSE & MAE")
    logs(f"RMSE: {rmse:.4f}")
    logs(f"MAE : {mae:.4f}")
    # plot -> logs -> save confusion matrix
    plot_confusion_matrix(con_mat, num_classes, path, nor=True)

    return clf, scaler


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
        model_name = "mlp_oneStep"

        # model locations
        model_dir = os.path.join(BASE, config["paths"][model_name]["model_path"])
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{model_name}_model.pkl") # path to saved model 
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl") # path to saved scaler
        history_path = os.path.join(model_dir, "history_data.pkl")

        # times
        sleep_time = metadata["model_param"]["sleep_time"]
    except Exception as e:
        logs(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # load and test saved model 
    load_result = load_model(model_path=model_path,scaler_path=scaler_path, history_path=history_path)
    if load_result is None:
        clf, clf_copy, scaler, test_data_old, test_labels_old = None, None, None, None, None
    else:
        clf, scaler, test_data_old, test_labels_old = load_result
        clf_copy = copy.deepcopy(clf)


    # while True:
    for _ in range(1):
        logs("Learning started.")            
        # --------------------------------------------------- load data ---------------------------------------------------
        # MLP Training on prometheus
        hours_loaded = metadata["data_param"]["hours_loaded"]
        end_time = datetime.now().replace(second=0, microsecond=0)
        start_time = end_time - timedelta(hours=hours_loaded)        
        train_data, train_labels, test_data, test_labels = load_prometheus(
            metadata=metadata,
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            # smote=True
        )


        # --------------------------------------------------- train model ---------------------------------------------------
        data_dir_path = os.path.join(BASE, config["paths"][model_name]["data_path"]) 
        clf, scaler = MLPClass_partial_train(
            clf=clf,
            scaler=scaler,
            kuber_data=[train_data, train_labels, test_data, test_labels],        
            metadata=metadata,
            path=data_dir_path
        )

        # --------------------------------------------------- save new model & scaler ---------------------------------------------------
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
            improvement = compare_models(
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
        
        
        time.sleep(sleep_time)



# start main
if __name__ == '__main__':
    main()
