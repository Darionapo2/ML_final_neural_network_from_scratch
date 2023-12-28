from typing import Callable
from utils import sigmoid

class Neuron:

    activation_value: float
    bias: float
    activation_function: Callable

    def __init__(self, activation_function: Callable = sigmoid):
        self.activation_function = activation_function
