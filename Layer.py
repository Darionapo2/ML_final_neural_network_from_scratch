from typing import Callable
from utils import sigmoid
from Neuron import Neuron


class Layer:
    neurons: list
    n: int
    activation_f: str

    def __init__(self, neurons_number: int, activation_function: Callable = sigmoid):
        self.n = neurons_number
        self.activation_f = str(activation_function)
        self.neurons = [Neuron(activation_function) for _ in range(self.n)]

    def __str__(self) -> str:
        return f'<Layer({self.n}, {self.activation_f})>'

    def feed(self, input_data: list[float]) -> list[float]:
        # pensarci
        return [self.activation_f(act_v) for act_v in input_data]
