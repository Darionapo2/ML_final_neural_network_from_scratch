import sys
from math import pow, e
import numpy as np
from Layer import *
import Layer

from numpy import typing


def sigmoid(x: float) -> float:
    return 1 / (1 + pow(e, -x))


def derived_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def identity(x: float) -> float:
    return x


def const_1(x: float) -> int:
    return 1


def softmax(x) -> np.array:
    exp_values = np.exp(x)
    exp_sum = np.sum(exp_values)
    return exp_values / exp_sum


def derived_softmax(x):
    s = softmax(x)
    return s * (1 - s)


activation_functions = {
    'sigmoid': sigmoid,
    'identity': identity,
    'softmax': softmax
}

derivatives = {
    'sigmoid': derived_sigmoid,
    'identity': const_1,
    'softmax': derived_softmax
}


def normal_distribution(n: int) -> np.ndarray:
    sigma = np.sqrt(2 / n)
    return np.random.normal(0, sigma, size=(n,))


def ones(n: int) -> list[int]:
    return [1] * n


def error(x1: np.ndarray, x2: np.ndarray) -> float:
    return sum(np.power(np.subtract(x1, x2), 2))


def get_error_vector(d: np.array, layer: Layer):
    pass


def get_difference(d: np.array, o: np.array):
    return np.subtract(d, o)
