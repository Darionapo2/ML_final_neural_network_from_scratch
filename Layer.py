from typing import Callable
from utils import sigmoid
from Neuron import Neuron
class Layer:

    neurons: list
    def __int__(self, number_neurons: int, activation_function: Callable = sigmoid):
        self.neurons = [Neuron(activation_function) for _ in range(number_neurons)]

