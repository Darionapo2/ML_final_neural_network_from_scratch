from typing import Callable

import numpy as np

from Neuron import Neuron
from utils import sigmoid


class Layer:
    neurons: list[Neuron]
    n: int
    activation_f: Callable

    def __init__(self, neurons_number: int, activation_function: Callable = sigmoid):
        self.n = neurons_number
        self.activation_f = activation_function
        self.neurons = [Neuron(self.activation_f) for _ in range(self.n)]

    def __str__(self) -> str:
        return f'<Layer({self.n}, {self.activation_f})>'

    def feed(self, input_data: np.array) -> bool:

        if len(input_data) != self.n:
            print('Error: invalid input data.')
            return False

        for i, neuron in enumerate(self.neurons):
            neuron.net = input_data[i]
            print('bias', i, ' ', neuron.bias)

        return True

    def activate(self):
        for neuron in self.neurons:
            neuron.out = neuron.activation_function(neuron.net)

    def get_output_values(self) -> list[float]:
        return [nr.out for nr in self.neurons]

    def get_weights(self) -> list[float]:
        return [nr.weights for nr in self.neurons]

    def input_weights_from_matrix(self, weights_m: np.array):
        for i, w_row in enumerate(weights_m):
            weights = w_row[1:]
            bias = w_row[0]
            print('weights: ', weights)
            print('bias: ', bias)

            self.neurons[i].weights = weights
            self.neurons[i].bias = bias
