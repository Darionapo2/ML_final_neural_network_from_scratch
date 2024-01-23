import sys
from math import pow, e
import numpy as np
from Layer import *
import Layer

from numpy import typing


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def derived_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def identity(x: float) -> float:
    return x


def const_1(x: float) -> int:
    return 1


def softmax(neuron_net: float, neurons: list) -> float:
    exp_sum = 0
    for nr in neurons:
        exp_sum += np.exp(nr.net)
    exp_value = np.exp(neuron_net)
    print(exp_value)
    return exp_value / exp_sum


def derives_softmax_output_l(neurons: list, diff: list) -> list:
    derivs = []
    for j, nr1 in enumerate(neurons):
        out_j = nr1.net
        dj = 0
        for i, nr2 in enumerate(neurons):
            out_i = nr2.net
            cond = 1 if i == j else out_i
            dj += (-1) * (diff[i] * out_i) * out_j * (cond - out_i)
        derivs.append(dj)
        # derivs.append(sum((-1) * diff[i] * nr2.net * nr1.net * (1 - nr2.net) if i == j else (-1) * diff[i] *
        # nr2.net * nr1.net * nr2.net for i, nr2 in enumerate(neurons)))
    return derivs


activation_functions = {
    'sigmoid': sigmoid,
    'identity': identity,
    'softmax': softmax,
}

derivatives = {
    'sigmoid': derived_sigmoid,
    'identity': const_1,
    'softmax': derives_softmax_output_l
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
