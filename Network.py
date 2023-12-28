from Neuron import Neuron
from Layer import Layer
class Network:
    input_layer: Layer
    output_layer: Layer
    hidden_layers: list[Layer]

    def __init__(self, network_shape: dict):
        self.input_layer = network_shape


    def backpropagate(self):
        pass

    def forwardpropagate(self):
        pass