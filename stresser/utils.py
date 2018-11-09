import numpy as np

def normalize_len(D, max_len):
    X, x = [], []
    for line in D:
        x = line[:max_len]
        while len(x) < max_len:
            x.append(2)
        X.append(x)
    return np.array(X, dtype=np.float32)