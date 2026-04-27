import numpy as np
from my_modules.printable_utils import logs

def _create_sequences(data, seq_length) -> tuple:
    '''Create sequence from data for LSTM learning'''
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_x = data[i:i + seq_length]
        seq_y = data[i + seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def data_split(
        data, 
        seq_length,
        feature_count = 1
    ) -> tuple:
    '''
    Split data for training / testing
    '''

    X, y = _create_sequences(data, seq_length)
    #  Reshape for model Input Format
    X = X.reshape((X.shape[0], seq_length, feature_count))
    #  Train/Test Split (Time-Aware)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logs(f"Training samples: {X_train.shape[0]}")
    logs(f"Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

# def data_split_multi(X, y, seq_length, split_ratio=0.8):
#     X_seqs, y_seqs = [], []
#     for i in range(len(X) - seq_length):
#         X_seqs.append(X[i:i+seq_length])
#         y_seqs.append(y[i+seq_length])

#     X_seqs = np.array(X_seqs)
#     y_seqs = np.array(y_seqs)

#     split_idx = int(len(X_seqs) * split_ratio)
#     X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
#     y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]

#     return X_train, X_test, y_train, y_test

# def data_split_multi(X, y, seq_length, split_ratio=0.8):
#     if X is None or y is None:
#         raise ValueError("X and y must not be None.")

#     if len(X) != len(y):
#         raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

#     if len(X) <= seq_length:
#         raise ValueError(
#             f"Not enough samples ({len(X)}) for seq_length={seq_length}."
#         )

#     X_seqs, y_seqs = [], []

#     for i in range(len(X) - seq_length):
#         X_seqs.append(X[i:i + seq_length])
#         y_seqs.append(y[i + seq_length])

#     X_seqs = np.asarray(X_seqs, dtype=np.float32)
#     y_seqs = np.asarray(y_seqs, dtype=np.float32)

#     split_idx = int(len(X_seqs) * split_ratio)

#     X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
#     y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]

#     logs(f"Training samples: {X_train.shape[0]}")
#     logs(f"Testing samples: {X_test.shape[0]}")
#     logs(f"X_train shape: {X_train.shape}")
#     logs(f"X_test shape: {X_test.shape}")
#     logs(f"y_train shape: {y_train.shape}")
#     logs(f"y_test shape: {y_test.shape}")

#     return X_train, X_test, y_train, y_test

def data_split_multi(X, y, seq_length, future_steps=5, split_ratio=0.8):
    if X is None or y is None:
        raise ValueError("X and y must not be None.")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

    if len(X) <= seq_length + future_steps - 1:
        raise ValueError(
            f"Not enough samples ({len(X)}) for seq_length={seq_length} and future_steps={future_steps}."
        )

    X_seqs, y_seqs = [], []

    for i in range(len(X) - seq_length - future_steps + 1):
        X_seqs.append(X[i:i + seq_length])

        # y has shape (N,1), so take next 5 rows and flatten to (5,)
        future_target = y[i + seq_length:i + seq_length + future_steps].reshape(-1)
        y_seqs.append(future_target)

    X_seqs = np.asarray(X_seqs, dtype=np.float32)
    y_seqs = np.asarray(y_seqs, dtype=np.float32)

    split_idx = int(len(X_seqs) * split_ratio)

    X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
    y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]

    logs(f"Training samples: {X_train.shape[0]}")
    logs(f"Testing samples: {X_test.shape[0]}")
    logs(f"X_train shape: {X_train.shape}")
    logs(f"X_test shape: {X_test.shape}")
    logs(f"y_train shape: {y_train.shape}")
    logs(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# def create_sequences_only(X, y, seq_length):
#     X_seqs, y_seqs = [], []

#     if X is None or y is None:
#         raise ValueError("X and y must not be None.")

#     if len(X) != len(y):
#         raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

#     if len(X) <= seq_length:
#         raise ValueError(f"Not enough samples ({len(X)}) for seq_length={seq_length}.")

#     for i in range(len(X) - seq_length):
#         X_seqs.append(X[i:i + seq_length])
#         y_seqs.append(y[i + seq_length])

#     X_seqs = np.asarray(X_seqs, dtype=np.float32)
#     y_seqs = np.asarray(y_seqs, dtype=np.float32)

#     logs(f"Test samples: {X_seqs.shape[0]}")
#     logs(f"X_test shape: {X_seqs.shape}")
#     logs(f"y_test shape: {y_seqs.shape}")

#     return X_seqs, y_seqs

def create_sequences_only(X, y, seq_length, future_steps=5):
    X_seqs, y_seqs = [], []

    if X is None or y is None:
        raise ValueError("X and y must not be None.")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

    if len(X) <= seq_length + future_steps - 1:
        raise ValueError(
            f"Not enough samples ({len(X)}) for seq_length={seq_length} and future_steps={future_steps}."
        )

    for i in range(len(X) - seq_length - future_steps + 1):
        X_seqs.append(X[i:i + seq_length])

        future_target = y[i + seq_length:i + seq_length + future_steps].reshape(-1)
        y_seqs.append(future_target)

    X_seqs = np.asarray(X_seqs, dtype=np.float32)
    y_seqs = np.asarray(y_seqs, dtype=np.float32)

    logs(f"Test samples: {X_seqs.shape[0]}")
    logs(f"X_test shape: {X_seqs.shape}")
    logs(f"y_test shape: {y_seqs.shape}")

    return X_seqs, y_seqs