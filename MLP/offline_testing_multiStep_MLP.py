import copy
from datetime import datetime
import os
import traceback
import numpy as np
import tqdm
import yaml
from sklearn.metrics import classification_report,accuracy_score, f1_score, precision_score, recall_score

from my_modules.model_utils import compare_multi_models, load_model, save_model, save_test_data_and_labels
from my_modules.printable_utils import  init_console, init_logs, logs, plot_confusion_matrix_mltiSteps
from my_modules.offline_testing_utils import compute_epoch_durations, load_synthetic_dataset, offline_data_preprocess, offline_data_preprocess_for_testing, split_dataset_by_interval
from main_multiStep_MLP import MultiOutClass_partial_train

# Intervals for testing:
INTERVALS = [
    # 364,    # 364 days -> 1 time (every year) 
    31,     # 31 days -> 12 times (every month)
    7,      # 7 days -> 52 times (every week)
    1,      # 1 day -> 364 times (every day)
]
INPUTS = ["cpu_features", "cpu_min_time_features", "multi_features"]
# INPUTS = ["cpu_features"]
# INPUTS = ["cpu_time_features"]
# INPUTS = ["cpu_min_time_features"]
# INPUTS = ["multi_features"]

def train(train_dataset):
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
        model_name = "mlp_multiStep"
        test_dir_path = os.path.join(BASE, "offline_tests_out")
        os.makedirs(test_dir_path, exist_ok=True)

        # initate logger
    except Exception as e:
        print(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)


    # load and test saved model 
    clf, clf_copy, scaler, scaler_copy, test_data_old, test_labels_old, last_model_mtime = None, None, None, None, None, None, None
    
    # ----------------------------------------------- load dataset ---------------------------------------------------
    for input_type in tqdm.tqdm(INPUTS, desc=f"Testing", leave=False):
        # test dataset on different intervals
        for interval in tqdm.tqdm(INTERVALS, desc=f"Testing:{input_type}", leave=False):
            # -------------------- create all needed dirs -----------------------------------
            input_interval_dir_path = os.path.join(test_dir_path, input_type, f"{interval}")
            data_dir_path = os.path.join(input_interval_dir_path, "data")
            model_dir = os.path.join(input_interval_dir_path, "model")
            model_iterations_dir = os.path.join(model_dir, "all_models")
            # create dirs
            os.makedirs(input_interval_dir_path, exist_ok=True)
            os.makedirs(data_dir_path, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(model_iterations_dir, exist_ok=True)

            # paths to files
            model_path = os.path.join(model_dir, "model.pkl") # path to saved model 
            scaler_path = os.path.join(model_dir, "scaler.pkl") # path to saved scaler
            history_path = os.path.join(model_dir, "history_data.pkl")

            # initae logs
            log_file_path = os.path.join(input_interval_dir_path, f"log.log")
            init_logs(log_file=log_file_path)

            # reset
            clf = clf_copy = scaler = scaler_copy = None
            test_data_old = test_labels_old = None
            last_model_mtime = None
            
            # logs("Learning started.")
            logs(f"Testing inputs: {metadata['data_param'][f'{input_type}']}...")
            logs(f"Testing interval: {interval} days...")    
    
            interval_dataset = split_dataset_by_interval(train_dataset, interval)

            for i, data in tqdm.tqdm(
                enumerate(interval_dataset),
                total=len(interval_dataset), 
                desc=f"Testing: {interval}", 
                leave=False,
        
            ):
                logs(f"Iteration {i + 1}/{len(interval_dataset)} for interval {interval} days...")
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
                                    scaler_copy = copy.deepcopy(scaler)
                                    logs("Model reloaded successfully")
                                last_model_mtime = current_mtime
                            except Exception as e:
                                logs(f"Error reloading model: {e}")
                    else:
                        logs("No model found -> making new one from scratch")
                        if clf is None and scaler is None:
                            logs(f"CLF and Scaler are None")
                        else:
                            logs(f"CLF or Scaler are not None!!")

                    # ---------------------------------------------------------- load data 
                    train_data, train_labels, test_data, test_labels = offline_data_preprocess(
                        dataset=data, 
                        metadata=metadata,
                        future_labels=metadata["model_param"][f"{model_name}_future_labels"],
                        numeric_features=metadata["data_param"][f"{input_type}"]
                    )

                    logs(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
                    logs(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")


                    if len(train_data) > 0:
                        # ---------------------------------------------------------- train model 
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
                        
                        # ---------------------------------------------------------- save copy
                        model_iteration_path = os.path.join(model_iterations_dir, f"model_{i+1}.pkl")
                        scaler_iteration_path = os.path.join(model_iterations_dir, f"scaler_{i+1}.pkl")
                        save_model(
                            clf=clf, 
                            model_path=model_iteration_path, 
                            scaler=scaler, 
                            scaler_path=scaler_iteration_path
                        )

                        # ---------------------------------------------------------- save new model & scaler 
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
                            scaler_copy = copy.deepcopy(scaler)
                            test_data_old, test_labels_old = save_test_data_and_labels(
                                test_database=[test_data, test_labels, test_data_old, test_labels_old], test_path=history_path)

                        else:
                            improvement = compare_multi_models(
                                            model_old=clf_copy, 
                                            model_new=clf,
                                            test_old_data= test_data_old,
                                            test_old_labels= test_labels_old,
                                            test_new_data=test_data, 
                                            test_new_labels=test_labels, 
                                            scaler_old=scaler_copy,
                                            scaler_new=scaler,
                                            n_deps=len(metadata["app_deployments"]),
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
                                scaler_copy = copy.deepcopy(scaler)
                                test_data_old, test_labels_old = save_test_data_and_labels(test_database=[test_data, test_labels, test_data_old, test_labels_old], test_path=history_path)

                            else:
                                logs("Old model is better or the same -> skip saving new model.")
                                # Restore clf to the best model
                                clf = copy.deepcopy(clf_copy)
                                scaler = copy.deepcopy(scaler_copy)
                    else:
                        logs("Not enough data...")

                except Exception as e:
                    logs(f"Error inside training loop...")
                    logs(f"Exception: {e}")
                    logs(traceback.format_exc())


def load_models(test_dir_path):
    try:
        models = {}
        for input_type in tqdm.tqdm(INPUTS, desc="Loading saved models", leave=False):
            for interval in tqdm.tqdm(INTERVALS, desc="Loading intervals", leave=False):
                model_dir = os.path.join(test_dir_path, input_type, f"{interval}", "model")
                model_path = os.path.join(model_dir, "model.pkl") # path to saved model 
                scaler_path = os.path.join(model_dir, "scaler.pkl") # path to saved scaler
                history_path = os.path.join(model_dir, "history_data.pkl")
                try:
                    load_result = load_model(
                        model_path=model_path, scaler_path=scaler_path, history_path=history_path)
                    if load_result is not None:
                        clf, scaler, _, _ = load_result
                        model_name = f"{input_type}_{interval}"
                        models[model_name] = [clf, scaler]
                except Exception as e:
                    logs(f"Could not found files for model with input type: {input_type} and interval {interval}...")

        return models            
    except Exception as e:
        logs(f"Error while loading models: {e}")


def test(test_dataset):

    # load metadata/paths to dirs
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, 'config.yaml')) as file:
            config = yaml.safe_load(file)

        metadata = config["metadata"]
        model_name = "mlp_multiStep"

        test_dir_path = os.path.join(BASE, "offline_tests_out")
        test_comparison_dir = os.path.join(test_dir_path, "comparisons")
        os.makedirs(test_comparison_dir, exist_ok=True)
        # output dirs pre grafy
        plot_dir = os.path.join(test_comparison_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # počet tried podľa tvojho binningu
        classes = metadata["data_param"]["class_values"]
        num_classes = len(classes)
        future_labels = metadata["model_param"][f"{model_name}_future_labels"]

        log_file_path = os.path.join(test_comparison_dir, "comparisons_log.log")
        init_logs(log_file=log_file_path)

    except Exception as e:
        print(f"Could not get relevant data: ({e}) -> terminating program.")
        raise  # lepšie než exit pri debugovaní

    try:
        models = load_models(test_dir_path=test_dir_path)
    except Exception as e:
        logs(f"Could not load models: {e}")

    for input_type in tqdm.tqdm(INPUTS, desc="Testing input types", leave=False):
        for interval in tqdm.tqdm(INTERVALS, desc="Testing intervals", leave=False):
            
            key = f"{input_type}_{interval}"
            if key not in models:
                logs(f"[WARN] Missing model key: {key} -> skipping")
                continue

            clf, scaler = models[key]
            
            input_interval_dir_path = os.path.join(test_dir_path, input_type, f"{interval}")
            train_log_file_path = os.path.join(input_interval_dir_path, f"log.log")
            log_file_path = os.path.join(test_comparison_dir, f"{input_type}_{interval}_finished_training_data.log")
            init_logs(log_file=log_file_path)


            # priprava test datasetu (celý súbor -> okná)
            X_test, y_test = offline_data_preprocess_for_testing(
                dataset=test_dataset,
                metadata=metadata,
                future_labels=future_labels,
                numeric_features=metadata["data_param"][input_type],
            )

            # scaling ak bol použitý pri tréningu
            # if scaler is not None:
            #     X_test = scaler.transform(X_test)
            n_deps = len(metadata["app_deployments"])
            n_numeric = X_test.shape[1] - n_deps

            X_num = X_test[:, :n_numeric].astype(float, copy=False)
            X_dep = X_test[:, n_numeric:].astype(float, copy=False)

            # voliteľná kontrola
            exp = getattr(scaler, "n_features_in_", None)
            if exp is not None and exp != X_num.shape[1]:
                raise ValueError(f"Scaler expects {exp} numeric features, but X_num has {X_num.shape[1]}")

            X_num = scaler.transform(X_num)
            X_test = np.hstack([X_num, X_dep])

            # predict
            y_pred = clf.predict(X_test)

            # sanity checks na tvary
            # pre multi-step: očakávam (N, future_labels)
            y_test = np.asarray(y_test)
            y_pred = np.asarray(y_pred)

            if future_labels == 1:
                # zjednoť na (N,1) aby bol rovnaký kód nižšie
                if y_test.ndim == 1:
                    y_test = y_test.reshape(-1, 1)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)

            if y_test.shape != y_pred.shape:
                logs(f"[ERROR] Shape mismatch for {key}: y_test {y_test.shape} vs y_pred {y_pred.shape}")
                continue

            # jedna presnosť pre test set (toto NIE JE train accuracy)
            # jedna presnosť pre test set (toto NIE JE train accuracy)
            acc = clf.score(X_test, y_test)
            logs(f"[{key}] Accuracy (on test file) = {acc:.4f}")

            # timestamp string (ak chceš unikátne názvy)
            timestamp = datetime.now()

            # --- NEW: zbieranie metrík pre priemery cez kroky ---
            step_accs = []
            step_f1_macro = []
            step_f1_weighted = []
            step_prec_macro = []
            step_rec_macro = []

            # report + confusion matrix pre každý krok dopredu
            for step in range(future_labels):
                y_true_s = y_test[:, step]
                y_pred_s = y_pred[:, step]

                logs(f"[{key}] Step {step+1}/{future_labels} - Classification Report:")
                class_report = classification_report(
                    y_true_s,
                    y_pred_s,
                    digits=4,
                    zero_division=0
                )
                logs(f"\n{class_report}")

                # --- NEW: metriky pre tento krok ---
                step_accs.append(accuracy_score(y_true_s, y_pred_s))
                step_f1_macro.append(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0))
                step_f1_weighted.append(f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0))
                step_prec_macro.append(precision_score(y_true_s, y_pred_s, average="macro", zero_division=0))
                step_rec_macro.append(recall_score(y_true_s, y_pred_s, average="macro", zero_division=0))

                # confusion matrix (num_classes x num_classes)
                con_mat = np.zeros((num_classes, num_classes), dtype=int)
                for actual, pred in zip(y_true_s, y_pred_s):
                    a = int(actual)
                    p = int(pred)
                    if 0 <= a < num_classes and 0 <= p < num_classes:
                        con_mat[a, p] += 1

                plot_confusion_matrix_mltiSteps(
                    con_mat=con_mat,
                    num_classes=num_classes,
                    path=plot_dir,
                    step=step + 1,
                    timestamp=timestamp,
                    nor=True,
                    name_addom=f"{input_type}_{interval}",
                    create_png=True,
                )

            # --- log hodnoty ---
            step1_f1 = step_f1_macro[0]
            step5_f1 = step_f1_macro[-1]

            logs(f"[{key}] data over {future_labels} steps:")
            logs("---------------------------")
            # f1 macro
            logs(f"  macro_f1_step_1      = {step1_f1:.4f}")
            logs(f"  macro_f1_step_last      = {step5_f1:.4f}")
            logs(f"  macro_f1_mean        = {np.mean(step_f1_macro):.4f} (min={np.min(step_f1_macro):.4f}, max={np.max(step_f1_macro):.4f})")
            logs(f"  subset_accuracy(clf.score) = {acc:.4f}")
            # --- čas pre tréning
            with open(train_log_file_path, "r", encoding="utf-8") as f:
                res = compute_epoch_durations(f.readlines())
                logs(f"  count: {res['count']}")
                logs(f"  total seconds: {res['total_sec']}")
                logs(f"  sec avg: {res['avg_sec']}")

            logs("---------------------------")
            logs(f"  step_accuracy_mean   = {np.mean(step_accs):.4f} (min={np.min(step_accs):.4f}, max={np.max(step_accs):.4f})")
            logs(f"  weighted_f1_mean     = {np.mean(step_f1_weighted):.4f} (min={np.min(step_f1_weighted):.4f}, max={np.max(step_f1_weighted):.4f})")
            logs(f"  macro_precision_mean = {np.mean(step_prec_macro):.4f}")
            logs(f"  macro_recall_mean    = {np.mean(step_rec_macro):.4f}")


def main():
    try:
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
    except Exception as e:
        logs(f"Error while testing: {e}")

# start main
if __name__ == '__main__':
    main()
