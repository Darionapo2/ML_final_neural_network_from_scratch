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


def normal_distribution(n):
    sigma = np.sqrt(2 / n)
    return np.random.normal(0, sigma, size = (n,))
