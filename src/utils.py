import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def modified_sigmoid(x: float) -> float:
    return 2 * sigmoid(x) - 1
