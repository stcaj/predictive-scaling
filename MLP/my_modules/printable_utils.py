import itertools
import logging
import os
import sys
import numpy as np

from datetime import datetime
from matplotlib import pyplot as plt

# internal logger reference
logger_file = None

def init_logs(log_file: str, level=logging.DEBUG):
    """
    Initialize logging for the program.

    Parameters:
    - log_file: path to the single log file
    - level: level of messages (default DEBUG to log everything)
    """
    global logger_file
    logger = logging.getLogger("programlogger_file")
    logger.setLevel(logging.DEBUG)   # accept all levels
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)       # write all levels
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

def logs(message: str):
    """
    Log a message using the program's logger.
    Automatically assigns INFO / WARNING / ERROR level
    based on message content.
    """
    if logger_file is None:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    else:
        text = message.lower()
        if "error" in text or "traceback" in text or "exception" in text:
            logger_file.error(message)
        elif "warn" in text:
            logger_file.warning(message)
        else:
            logger_file.info(message)
     
def plot_confusion_matrix(
        con_mat, 
        num_classes: int, 
        path: str, 
        nor = False
    ) -> None:
    '''
    plot -> save -> show confusion matrix
    '''
    plt.figure()
    logs(f"Confusion Matrix:")
    logs(f"\n{con_mat}")
    
    # noramlize the matrix
    if nor:
        con_mat = con_mat.astype(float) / con_mat.sum(axis=0)
        # con_mat = np.divide(
        #     con_mat.astype(float),
        #     con_mat.sum(axis=0),
        #     out=np.zeros_like(con_mat, dtype=float),
        #     where=con_mat.sum(axis=0)!=0
        # )
    '''
    if nor:
        row_sums = con_mat.sum(axis=1, keepdims=True)
        con_mat = np.divide(con_mat, row_sums, where=row_sums != 0)
    '''


    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Maticia zámien")
    plt.colorbar()
    plt.xticks(np.arange(num_classes), np.arange(num_classes))
    plt.yticks(np.arange(num_classes), np.arange(num_classes))
    fmt = '.2f' if nor else '.0f'
    threshhold = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, format(con_mat[i, j], fmt).replace('.', ','),
                 horizontalalignment="center",
                 color="white" if con_mat[i, j] > threshhold else "black")
    plt.tight_layout()
    plt.xlabel('Skutočná trieda')
    plt.ylabel('Predikovaná trieda')

    os.makedirs(path, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"matica_zamien_{timestamp}.png"
    plt.savefig(os.path.join(path, file_name), bbox_inches='tight')
    plt.close()
    logs(f"Matica uložená ako '{file_name}'")
    # plt.show()



def plot_confusion_matrix_mltiSteps(
        con_mat, 
        num_classes: int, 
        path: str,
        step: int,
        timestamp: str,
        nor = False,
        name_addom = None,
        debug = False,
        create_png = False,
    ) -> None:
    '''
    plot -> save -> show confusion matrix for evcery step in multi step model
    '''
    timestamp = timestamp.strftime("%Y%m%d_%H-%M-%S")
    if debug:
        logs(f"Confusion Matrix for prediction step {step}:")
        logs(f"\n{con_mat}")
    

    if create_png:
        plt.figure()
        # noramlize the matrix
        # if nor:
            # con_mat = con_mat.astype(float) / con_mat.sum(axis=0)
        if nor:
            axis0 = con_mat.sum(axis=0)
            axis0[axis0 == 0] = 1
            con_mat = con_mat.astype(float) / axis0

        '''
        if nor:
            row_sums = con_mat.sum(axis=1, keepdims=True)
            con_mat = np.divide(con_mat, row_sums, where=row_sums != 0)
        '''

        plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Matica zámien - Krok {step}")
        # plt.colorbar()
        plt.xticks(np.arange(num_classes), np.arange(num_classes))
        plt.yticks(np.arange(num_classes), np.arange(num_classes))
        fmt = '.2f' if nor else '.0f'
        threshhold = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, format(con_mat[i, j], fmt).replace('.', ','),
                    horizontalalignment="center",
                    color="white" if con_mat[i, j] > threshhold else "black")
        plt.tight_layout()
        plt.xlabel('Skutočná trieda')
        plt.ylabel('Predikovaná trieda')

        os.makedirs(path, exist_ok=True)
        name_start = f"{name_addom}_" if name_addom != None else ""
        plot_name = f"{name_start}krok_{step}_matica_zamien_{timestamp}.png"
        plt.savefig(os.path.join(path, plot_name), bbox_inches='tight')
        logs(f"Matica uložená ako '{plot_name}'")
        # plt.show()
        plt.close()
