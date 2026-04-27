import os
from sklearn.metrics import classification_report
import yaml
from datetime import datetime
from my_modules.fastStorage_loader import load_fastStorage_multi
from my_modules.printable_utils import init_logs, logs, plot_confusion_matrix_mltiSteps

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def MultiOutClass_partial_train(
        train_data, 
        train_labels, 
        test_data, 
        test_labels,
        metadata,
        plot_path
):
    """
    Incremental training of MultiOutput MLPClassifier with partial_fit.

    Ensures each batch contains samples from all classes and handles multi-step labels.
    """
    # Metadata
    epochs = metadata["model_param"].get("epochs", 10)
    batch_size = metadata["model_param"].get("batch_size", 32)
    hidden_layer_sizes = metadata["model_param"].get("hidden_layer_sizes", (100,50))
    max_iter = metadata["model_param"].get("max_iter", 200)
    random_state = metadata["model_param"].get("random_state", 42)
    num_classes = len(metadata["fastStorage_param"]["class_values"])

    # Preprocess
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Handle multi-output label shape
    if len(train_labels.shape) == 1:
        train_labels = train_labels.reshape(-1, 1)
        test_labels = test_labels.reshape(-1, 1)

    n_outputs = train_labels.shape[1]
    classes = [np.arange(num_classes) for _ in range(n_outputs)]

    # MLP Classifier
    clf = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            max_iter=max_iter, 
            random_state=random_state
        )
    )

    # Training loop
    for epoch in range(epochs):
        logs(f"Epoch {epoch + 1}/{epochs}...")

        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]

        for start in range(0, len(train_data), batch_size):
            end = start + batch_size
            X_batch = train_data_shuffled[start:end]
            y_batch = train_labels_shuffled[start:end].astype(int)  # force int

            if epoch == 0 and start == 0:
                clf.partial_fit(X_batch, y_batch, classes=classes)
            else:
                clf.partial_fit(X_batch, y_batch)
            # scale aggain for
    
    
    logs("Train accuracy = {:.4f}".format(clf.score(train_data, train_labels)))
    logs("Test accuracy  = {:.4f}".format(clf.score(test_data, test_labels)))

    # Confusion matrix (one per output)
    test_pred = clf.predict(test_data)
    timestamp = datetime.now()

    # Classification report
    # logs("Global Classification Report:")
    # logs(classification_report(test_labels, test_pred, digits=4))

    for step in range(n_outputs):
        logs(f"Classification Report for step {step}:\n" + classification_report(test_labels[:, step], test_pred[:, step], digits=4))
        
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


def main():
    # load metadata/paths to dirs
    try:
        # load config.yaml
        BASE = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE, CONFIG_PATH)) as file:
            config = yaml.safe_load(file)

        # metadata
        metadata = config["metadata"]
        model_name = "mlp_multiStep"
        
        # extract metadata
        SAMPLE_SIZE_AVERAGE = metadata["model_param"]["sample_history"] 
        class_values = metadata["fastStorage_param"]["class_values"]
        future_labels = metadata["model_param"]["mlp_multiStep_future_labels"]
        skipped_steps = metadata["model_param"]["skipped_steps"]
    
        # initate logger
        init_logs(log_file="MLP/logs/mlp_model_training_fastStorage.log")
        
        num_classes = len(class_values)
        logs(f"Starting partial_fit training with [{num_classes}] classes...")

        # graphs locations
        plot_path = os.path.join(BASE, config["paths"]["logs"]["log_path"])
        os.makedirs(plot_path, exist_ok=True)
    except Exception as e:
        print(f"Could not get relevant data: ({e}) -> terminating program.")
        exit(-1)

    # MLP Training
    train_data, train_labels, test_data, test_labels = load_fastStorage_multi(
        class_values=class_values, future_labels=future_labels, SAMPLE_SIZE_AVERAGE=SAMPLE_SIZE_AVERAGE, skipped_steps=skipped_steps)
    
    if len(train_data) == 0:
        logs("ERROR: No samples loaded from fastStorage directory")
        exit(-1)

    _, _ = MultiOutClass_partial_train(
        train_data=train_data, 
        train_labels=train_labels, 
        test_data=test_data, 
        test_labels=test_labels,
        metadata=metadata,
        plot_path=plot_path
    )



# main
if __name__ == '__main__':
    main()
