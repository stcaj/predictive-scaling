import glob
import csv
import os
from statistics import mean, stdev
from datetime import datetime

from my_modules.printable_utils import logs

# Paths
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
FASTSTORAGE_DIR = os.path.join(BASE, "fastStorage")
OUTPUT_DIR = os.path.join(BASE, "output")

def classification_of_fastStorage(class_values: list, SAMPLE_SIZE_AVERAGE = 10):
    '''
    # SAMPLE_SIZE_AVERAGE - Number of entries to average
    '''
    logs(f"Starting classification...")
    input_pattern = os.path.join(FASTSTORAGE_DIR, "*.csv")
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # goes through files
    for fname in glob.glob(input_pattern):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            next(reader)  # skip header

            basename = os.path.basename(fname)
            output_file_path = os.path.join(output_dir, basename)

            with open(output_file_path, 'w', newline='') as outfh:
                writer = csv.writer(outfh, delimiter=';')
                writer.writerow([
                    "timestamp", "hour", "minute", "day_of_week", "month",
                    "cpu_avg", "mem_avg",
                    "class_num"
                ])

                buffers = {
                    "timestamp": [],
                    "cpu": [],
                    "mem_percent": []
                }

                for row in reader:
                    if len(row) < 7:
                        continue  # skip incomplete rows

                    try:
                        timestamp = int(row[0])
                        cpu_usage = float(row[4])
                        mem_capacity = float(row[5])
                        mem_usage = float(row[6])

                        mem_usage_percent = (mem_usage / mem_capacity) * 100.0 if mem_capacity > 0 else 0.0
                    except (ValueError, ZeroDivisionError):
                        continue

                    buffers["timestamp"].append(timestamp)
                    buffers["cpu"].append(cpu_usage)
                    buffers["mem_percent"].append(mem_usage_percent)

                    if len(buffers["timestamp"]) >= SAMPLE_SIZE_AVERAGE:
                        timestamp_avg = mean(buffers["timestamp"])
                        # dt = datetime.fromtimestamp(timestamp_avg)
                        dt = datetime.fromtimestamp(timestamp_avg)

                        hour = dt.hour
                        minute = dt.minute
                        day_of_week = dt.weekday()
                        month = dt.month

                        cpu_avg = mean(buffers["cpu"])
                        mem_avg = mean(buffers["mem_percent"])

                        # Classification based on cpu_perc
                        class_num = None
                        for cpu_class_perc in class_values:
                            if cpu_class_perc >= cpu_avg:
                                class_num = class_values.index(cpu_class_perc)
                                break

                        if class_num is None:  # if more than 100% -> last class
                            class_num = len(class_values) - 1
                            
                        # # Classification based on cpu_avg
                        # if cpu_avg < 2.5:
                        #     class_num = 0
                        # elif cpu_avg < 5.0:
                        #     class_num = 1
                        # elif cpu_avg < 10.0:
                        #     class_num = 2
                        # elif cpu_avg < 20.0:
                        #     class_num = 3
                        # elif cpu_avg < 40.0:
                        #     class_num = 4
                        # else:
                        #     class_num = 5

                        writer.writerow([
                            timestamp_avg, 
                            hour, minute, day_of_week, month,
                            cpu_avg, mem_avg,
                            class_num
                        ])

                        # Clear buffers
                        for key in buffers:
                            buffers[key].clear()

    logs(f"Classifying done...")




def classification_of_fastStorage_with_rolling_window(SAMPLE_SIZE_AVERAGE = 10):
    '''
    # SAMPLE_SIZE_AVERAGE - Number of entries to average
    '''
    logs(f"Starting classification with rolling window...")
    input_pattern = os.path.join(BASE, "fastStorage", "*.csv")
    output_dir = os.path.join(BASE, "output_rolling")    
    os.makedirs(output_dir, exist_ok=True)
    
    # goes through files
    for fname in glob.glob(input_pattern):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            next(reader)  # skip header

            basename = os.path.basename(fname)
            output_file_path = os.path.join(output_dir, basename)

            with open(output_file_path, 'w', newline='') as outfh:
                writer = csv.writer(outfh, delimiter=';')
                writer.writerow([
                    "hour", "minute", "day_of_week", "month",
                    "cpu_avg", "cpu_std", "cpu_diff", "cpu_min", "cpu_max",
                    "mem_avg", "mem_std", "mem_diff",
                    "class_num"
                ])

                buffers = {
                    "timestamp": [],
                    "cpu": [],
                    "mem_percent": []
                }
                
                for row in reader:
                    if len(row) < 7:
                        continue  # skip incomplete

                    try:
                        timestamp = int(row[0])
                        cpu_usage = float(row[4])
                        mem_capacity = float(row[5])
                        mem_usage = float(row[6])
                        mem_usage_percent = (mem_usage / mem_capacity) * 100.0 if mem_capacity > 0 else 0.0
                    except (ValueError, ZeroDivisionError):
                        continue

                    # Append new values
                    buffers["timestamp"].append(timestamp)
                    buffers["cpu"].append(cpu_usage)
                    buffers["mem_percent"].append(mem_usage_percent)

                    # Keep window size
                    if len(buffers["timestamp"]) >= SAMPLE_SIZE_AVERAGE:

                        # --- Calculate features ---
                        timestamp_avg = mean(buffers["timestamp"])
                        dt = datetime.fromtimestamp(timestamp_avg)
                        hour = dt.hour
                        minute = dt.minute
                        day_of_week = dt.weekday()
                        month = dt.month

                        # CPU
                        cpu_avg = mean(buffers["cpu"])
                        cpu_std = stdev(buffers["cpu"]) if len(buffers["cpu"]) > 1 else 0
                        cpu_diff = buffers["cpu"][-1] - buffers["cpu"][0]
                        cpu_min = min(buffers["cpu"])
                        cpu_max = max(buffers["cpu"])

                        # Memory
                        mem_avg = mean(buffers["mem_percent"])
                        mem_std = stdev(buffers["mem_percent"]) if len(buffers["mem_percent"]) > 1 else 0
                        mem_diff = buffers["mem_percent"][-1] - buffers["mem_percent"][0]

                        # --- Classification based on CPU avg ---
                        if cpu_avg < 2.5:
                            class_num = 0
                        elif cpu_avg < 5.0:
                            class_num = 1
                        elif cpu_avg < 10.0:
                            class_num = 2
                        elif cpu_avg < 20.0:
                            class_num = 3
                        elif cpu_avg < 40.0:
                            class_num = 4
                        else:
                            class_num = 5

                        # --- Write output row ---
                        writer.writerow([
                            hour, minute, day_of_week, month,
                            cpu_avg, cpu_std, cpu_diff, cpu_min, cpu_max,
                            mem_avg, mem_std, mem_diff,
                            class_num
                        ])
                    
                        # Clear buffers
                        for key in buffers:
                            buffers[key].clear()

    logs(f"Classification done...")


if __name__ == '__main__':
    classification_of_fastStorage()
