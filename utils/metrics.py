import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return ((predictions - targets) ** 2).mean()
