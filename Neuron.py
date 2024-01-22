from typing import Callable
from utils import sigmoid
import numpy as np


class Neuron:
    net: float
    out: float
    bias: float
    activation_function: Callable
    weights: np.array

    accumulate_change: np.array
    accumulate_change_bias: float
    previous_change: np.array
    previous_change_bias: float

    def __init__(self, activation_function: Callable = sigmoid):
        self.activation_function = activation_function
        self.bias = 0
        self.accumulate_change_bias = 0
        self.previous_change_bias = 0

    def __str__(self):
        return f'<Neuron({str(self.activation_function)}, value = {self.net}, bias = {self.bias})>'

    def get_weights(self):
        return self.weights

    def set_bias(self, value):
        self.bias = value
