from datetime import datetime
import glob
import csv
import os

from my_modules.printable_utils import logs

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "data")
INPUT_DIR = os.path.join(BASE, "output")
OUTPUT_DIR = os.path.join(BASE, "ml_data")
MULTI_OUTPUT_DIR = os.path.join(BASE, "multi_ml_data")

def preprocessing_of_fastStorage(future_labels=1, features_multiplier=10, skipped_steps=8, features=7):
    
    # features_multiplier: 10  # number of time steps in the input sequence
    # features: 7 # number of features
    # future_labels: 1 # number of future steps to predict
    # skipped_steps: 8 # number of future steps to skip (time for pods to init)

    path = os.path.join(INPUT_DIR, "*.csv")
    if future_labels == 1:
        output_path = OUTPUT_DIR
    else:
        output_path = MULTI_OUTPUT_DIR


    logs(f"Starting preprocessing...")
    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            next(reader)

            basename = os.path.basename(fname)
            os.makedirs(output_path, exist_ok=True)
            output = open(os.path.join(output_path, basename), "w+")

            csv_file_list = []

            for row in reader:
                timestamp = float(row[0])
                hour = int(row[1])
                minute = int(row[2])
                day_of_week = int(row[3])
                month = int(row[4])
                cpu_avg = float(row[5])
                mem_avg = float(row[6])
                class_num = int(row[7])
                csv_file_list.append([timestamp, hour, minute, day_of_week, month, cpu_avg, mem_avg, class_num])

            i = 0
            # while i < len(csv_file_list) - features_multiplier - label_steps:
            while i < len(csv_file_list) - features_multiplier - skipped_steps - future_labels:

                # Write input sequence
                for j in range(i, i + features_multiplier):
                    fields = csv_file_list[j][:features]  # all columns
                    output.write(";".join(str(f) for f in fields) + ";")


                # Write multi-step output with skipped steps
                labels = [csv_file_list[j + skipped_steps + k][features] for k in range(future_labels)]
                next_label = ";".join(map(str, labels))
                
                # next_label = csv_file_list[i + features_multiplier + skipped_steps][features]
                output.write(f"{next_label}\n")

                i += 1
                
    logs(f"Preprocessing done...")


def preprocessing_of_fastStorage_with_rolling_window(features_multiplier=10, future_labels=1, skipped_steps=8, features=12):


    # features_multiplier = 10  # number of time steps in the input sequence
    # future_labels = 5 # number of future steps to predict
    # skipped_steps = 1 # number of future steps to skip (time for pods to init)
    # features = 12 # number of features

    path = os.path.join(BASE, "output_rolling", "*.csv")
    output_path = os.path.join(BASE, "ml_rolling_data")
    logs(f"Starting preprocessing...")
    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            next(reader)

            basename = os.path.basename(fname)
            os.makedirs(output_path, exist_ok=True)
            output = open(os.path.join(output_path, basename), "w+")

            csv_file_list = []

            for row in reader:
                hour = int(row[0])
                minute = int(row[1])
                day_of_week = int(row[2])
                month = int(row[3])
                cpu_avg = float(row[4])
                cpu_std = float(row[5])
                cpu_diff = float(row[6])
                cpu_min = float(row[7])
                cpu_max = float(row[8])
                mem_avg = float(row[9])
                mem_std = float(row[10])
                mem_diff = float(row[11])
                class_num = int(row[12])
                csv_file_list.append([hour, minute, day_of_week, month,
                        cpu_avg, cpu_std, cpu_diff, cpu_min, cpu_max,
                        mem_avg, mem_std, mem_diff,
                        class_num])

            i = 0
            # while i < len(csv_file_list) - features_multiplier - label_steps:
            while i < len(csv_file_list) - features_multiplier - skipped_steps - future_labels:

                # Write input sequence
                for j in range(i, i + features_multiplier):
                    fields = csv_file_list[j][:features]  # all columns
                    output.write(";".join(str(f) for f in fields) + ";")


                # Write multi-step output with skipped steps
                # labels = [csv_file_list[j + skipped_steps + k][features] for k in range(future_labels)]
                # next_label = ";".join(map(str, labels))
                next_label = csv_file_list[i + features_multiplier + skipped_steps][features]
                output.write(f"{next_label}\n")

                i += 1
                
    logs(f"Preprocessing done...")


if __name__=='__main__':
    preprocessing_of_fastStorage()