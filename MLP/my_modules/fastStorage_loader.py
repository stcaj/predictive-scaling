import csv
from datetime import datetime
import glob
import os
from typing import Counter
import numpy as np
from imblearn.over_sampling import SMOTE

from my_modules.fastStorage_data_classification import classification_of_fastStorage
from my_modules.fastStorage_data_preprocess import preprocessing_of_fastStorage
from my_modules.printable_utils import logs

BASE = os.path.dirname(os.path.abspath(__file__))
FASTSTORAGE_DIR = os.path.join(BASE, "..", "..", "data")
ML_DATA_DIR = os.path.join(FASTSTORAGE_DIR, "ml_data")
MULTI_ML_DATA = os.path.join(FASTSTORAGE_DIR, "multi_ml_data")
ML_ROLLING_DATA_DIR = os.path.join(FASTSTORAGE_DIR, "ml_rolling_data")

# load correct data for ML with one step
def load_data_one_step():
    path = os.path.join(ML_DATA_DIR, "*.csv")
    # path = os.path.join(ML_ROLLING_DATA_DIR, "*.csv")

    data = []
    labels = []
    logs(f"loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype = float)
                data.append(r[:-1])
                labels.append(r[-1])
    
    data = np.array(data)
    labels = np.array(labels)

    if len(data) == 0:
        logs(f"Data is empty!")
        exit(-1)

    n = int(len(data) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]

    logs(f"finished loading data...")
    return train_data, train_labels, test_data, test_labels


# load correct data for ML (only CPU)
def load_data_one_step_Smote():
    path = os.path.join(ML_DATA_DIR, "*.csv")

    data = []
    labels = []
    logs(f"loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype = float)
                data.append(r[:-1])
                labels.append(r[-1])
    
    data = np.array(data)
    labels = np.array(labels)

    n = int(len(data) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]

    logs(f"finished loading data...")

    # Apply SMOTE ONLY to training data
    # smote = SMOTE(random_state=42, k_neighbors=3)
    smote = SMOTE(random_state=42, sampling_strategy={1: 60000, 2: 60000})
    train_data_resampled, train_labels_resampled = smote.fit_resample(train_data, train_labels)
    return train_data_resampled, train_labels_resampled, test_data, test_labels


# load correct data for ML
def load_data_multi_step():
    future_labels = 5 #1 # number of steps
    
    path = os.path.join(ML_DATA_DIR, "*.csv")
    path = os.path.join(MULTI_ML_DATA, "*.csv")

    data = []
    labels = []
    logs(f"loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype = float)
                data.append(r[:-future_labels])
                labels.append(r[-future_labels:])
    
    data = np.array(data)
    labels = np.array(labels)

    n = int(len(data) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]

    logs(f"finished loading data...")
    return train_data, train_labels, test_data, test_labels


# show ho many instances of class are there
def count_classes(path: str) -> None:
    '''
    count how many of instances of all classes are there
    '''
    logs(f"counting labels...")
    label_counts = Counter()

    for fname in glob.glob(path):
        with open(fname, 'r') as f:

            reader = csv.reader(f, delimiter=';')
            next(reader)  # skip header

            for row in reader:
                try:
                    label = int(row[-1])  # last column
                    label_counts[label] += 1
                except:
                    logs(f"Bad row in {fname}: {row}")

    for label in sorted(label_counts):
        logs(f"Class {label}: {label_counts[label]} samples")
    
    return len(label_counts)



def load_fastStorage(SAMPLE_SIZE_AVERAGE: int, class_values: list):
    '''
    classify and preprocess data
    load and return preprocessed data
    '''
    # classification of data and preprocess
    classification_of_fastStorage(class_values=class_values ,SAMPLE_SIZE_AVERAGE=SAMPLE_SIZE_AVERAGE)
    preprocessing_of_fastStorage()

    # --- count how many classes are represented in files
    csv_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","fastStorageData","output", "*.csv")  # output path
    count_classes(path=csv_files_path)

    return load_data_one_step()



def load_fastStorage_multi(
        SAMPLE_SIZE_AVERAGE: int, 
        class_values: list, 
        future_labels: int, 
        skipped_steps: int
    ):
    '''
    classify -> preprocess -> return preprocessed data

    Parameters:
    - SAMPLE_SIZE_AVERAGE: 
    - class_values:
    - future_labels:
    - skipped_steps:

    Returns:
    - Preprocesses data for train / test
    '''
    # classification of data and preprocess
    classification_of_fastStorage(class_values=class_values, SAMPLE_SIZE_AVERAGE=SAMPLE_SIZE_AVERAGE)
    preprocessing_of_fastStorage(future_labels=future_labels, skipped_steps=skipped_steps)

    # --- count how many classes are represented in files
    csv_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fastStorageData","output", "*.csv")  # output path
    count_classes(path=csv_files_path)

    return load_data_multi_step()