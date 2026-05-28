import os
import numpy as np

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing
def min_max_normalize(x):
    result = (x - np.min(x)) / (np.max(x) - np.min(x))
    return result