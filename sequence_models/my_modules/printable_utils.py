from datetime import datetime
import logging
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# internal logger reference
logger_file = None

def init_logs(log_file: str, level=logging.DEBUG):
    global logger_file
    logger = logging.getLogger("programlogger_file")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger_file = logger

def init_console(level=logging.DEBUG):
    """
    Initialize logger to write ONLY to console.
    This will remove any existing handlers (including file handlers).
    """
    global logger_file

    logger = logging.getLogger("programlogger_file")
    logger.setLevel(level)
    logger.propagate = False

    # vymaže všetky existujúce handlery
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    logger_file = logger

def logs(message, level="info", console=True):
    """
    Log a message using the program's logger.
    Automatically assigns INFO / WARNING / ERROR level
    based on message content.
    """
    if logger_file is None:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    else:
        if level == "error":
            logger_file.error(message)
        elif level == "warn":
            logger_file.warning(message)
        else:
            logger_file.info(message)
    # else:
    #     text = message.lower()
    #     if "error" in text or "traceback" in text or "exception" in text:
    #         logger_file.error(message)
    #     elif "warn" in text:
    #         logger_file.warning(message)
    #     else:
    #         logger_file.info(message)


def create_graphs(
        predictions_rescaled,
        y_test_rescaled, 
        dir_path, 
        timestamp, 
        start_pos, 
        end_pos,
        name = "default"
    ) -> None:
    '''
    Makes time grapf and histogram

    Parameters
    ----------
    - predictions_rescaled: prediction values
    - y_test_rescaled: true values
    - dir_path: path where the graphs should be stored
    - timestamp: time when the training stoped
    - start_pos: start position of time graph
    - end_pos: end position of time graph
    - name : name of the model
    '''
    # file path for saving data
    os.makedirs(os.path.join(dir_path, "data"), exist_ok=True)
    timestamp = timestamp.replace("-", "").replace(" ", "_").replace(":", "-")

    time_plot_path = os.path.join(dir_path, "data", "time_figure")
    histogram_plot_path = os.path.join(dir_path, "data", "histogram_figure")

    # histogram of errors
    errors = predictions_rescaled - y_test_rescaled
    mean_error = np.mean(errors)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=60, color="#109217", edgecolor='black', label='Histogram chýb')  
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Priemerná chyba: {mean_error:.4f}')
    plt.title("Distribúcia chýb predikcie")
    plt.xlabel("Chyba predikcie")
    plt.ylabel("Frekvencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(histogram_plot_path + f'_{name}_{timestamp}.png', bbox_inches='tight')
    logs(f"Histogram saved as '{histogram_plot_path}_{name}_{timestamp}.png'")
    # plt.show()


    # time graf
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_rescaled[start_pos:end_pos], label='Skutočné hodnoty')
    plt.plot(predictions_rescaled[start_pos:end_pos], label='Predikcia')

    plt.title('Predikcia využitia CPU')
    plt.xlabel('Čas')
    plt.ylabel('CPU využitie')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(time_plot_path + f'_{name}_{timestamp}.png', bbox_inches='tight')
    logs(f"Time graph saved as '{time_plot_path}_{name}_{timestamp}.png'")
    # plt.show()


def check_for_nans(name, arr):
    if np.isnan(arr).any():
        idx = np.argwhere(np.isnan(arr))
        logs(f"[NaN DETECTED] in {name} at positions: {idx[:10]} (showing first 10)")
    if np.isinf(arr).any():
        idx = np.argwhere(np.isinf(arr))
        logs(f"[Inf DETECTED] in {name} at positions: {idx[:10]} (showing first 10)")
    logs(f"{name} -> shape={arr.shape}, min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}")
    


