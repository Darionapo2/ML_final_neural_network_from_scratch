from typing import Callable
from utils import sigmoid
from Neuron import Neuron


class Layer:
    neurons: list

    def __init__(self, neurons_number: int, activation_function: Callable = sigmoid):
        self.neurons = [Neuron(activation_function) for _ in range(neurons_number)]
