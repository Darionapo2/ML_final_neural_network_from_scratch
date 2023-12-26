from Neuron import Neuron
class Network:
    input_neurons = []

    def __init__(self, input_neurons_number: int, hidden_layers_number: int, output_neurons_number: int):
        self.fisrt_layer_neurons = [Neuron(0) for _ in range(input_neurons_number)]