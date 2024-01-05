from typing import Callable
from utils import sigmoid
from Neuron import Neuron


class Layer:
    neurons: list
    n: int
    activation_f: Callable

    def __init__(self, neurons_number: int, activation_function: Callable = sigmoid):
        self.n = neurons_number
        self.activation_f = activation_function
        self.neurons = [Neuron(self.activation_f) for _ in range(self.n)]

    def __str__(self) -> str:
        return f'<Layer({self.n}, {self.activation_f})>'

    def feed(self, input_data: list[float]) -> bool:

        if len(input_data) != self.n:
            print('Error: invalid input data.')
            return False

        for i, neuron in enumerate(self.neurons):
            neuron.net = input_data[i]

        return True

    def activate(self):
        for neuron in self.neurons:
            neuron.out = neuron.activation_function(neuron.net)

    def get_output_values(self) -> list[float]:
        return [nr.out for nr in self.neurons]
