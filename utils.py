from math import pow, e
import numpy as np
from numpy import typing


def sigmoid(x: float) -> float:
    return 1 / (1 + pow(e, -x))


def derived_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def identity(x: float) -> float:
    return x


activation_functions = {
    'sigmoid': sigmoid,
    'identity': identity
}


def normal_distribution(n: int) -> np.ndarray:
    sigma = np.sqrt(2 / n)
    return np.random.normal(0, sigma, size = (n,))


def ones(n: int) -> list[int]:
    return [1] * n


def distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return sum(np.subtract(np.power(x1, 2), np.power(x2, 2)))
