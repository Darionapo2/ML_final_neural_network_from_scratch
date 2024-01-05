from typing import Callable
from utils import sigmoid


class Neuron:
    net: float
    out: float
    bias: float
    activation_function: Callable
    weights = list[float]

    def __init__(self, activation_function: Callable = sigmoid):
        self.activation_function = activation_function

        # to remove
        self.bias = 0

    def __str__(self):
        return f'<Neuron({str(self.activation_function)}, value = {self.net}, ' \
               f'bias = {self.bias})>'

    def get_weights(self):
        return self.weights
