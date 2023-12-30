from typing import Callable
from utils import sigmoid


class Neuron:
    activation_value: float
    bias: float
    activation_function: Callable

    def __init__(self, activation_function: Callable = sigmoid):
        self.activation_function = activation_function

    def __str__(self):
        return f'<Neuron({str(self.activation_function)}, value = {self.activation_value}, ' \
               f'bias = {self.bias})>'
