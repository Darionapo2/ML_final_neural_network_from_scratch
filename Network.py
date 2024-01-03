from Neuron import Neuron
from Layer import Layer
from utils import sigmoid, activation_functions, normal_distribution
import numpy as np


class Network:
    input_layer: Layer
    output_layer: Layer
    hidden_layers: list[Layer] = []

    def __init__(self, network_shape: list[dict]):
        input_layer_shape = network_shape[0]
        k = input_layer_shape['neurons_number']
        self.input_layer = Layer(k, activation_functions['identity'])

        for i, layer_shape in enumerate(network_shape[1:]):

            n = layer_shape['neurons_number']
            selected_act_f = layer_shape['activation_f']
            if selected_act_f in activation_functions:
                act_f = activation_functions[selected_act_f]
            else:
                print(f'Error: Layer n. {i} activation function not avaiable. '
                      'Sigmoid is used instead.')
                act_f = sigmoid

            if i == len(network_shape) - 2:
                self.output_layer = Layer(n, act_f)
            else:
                self.hidden_layers.append(Layer(n, act_f))

        self.set_weights()

    def __str__(self):
        str_hls = [str(hl) for hl in self.hidden_layers]
        return f'<Network({str(self.input_layer)}, {", ".join(str_hls)}, {str(self.output_layer)})>'

    def set_weights(self):
        n_layer_before = 0

        for layer in self.hidden_layers + [self.output_layer]:
            for neuron in layer.neurons:
                if n_layer_before == 0:
                    n_layer_before = self.input_layer.n
                neuron.weights = normal_distribution(n_layer_before)
            n_layer_before = layer.n

    def forwardpropagate(self, input_data: list[float]):

        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for i, layer in enumerate(all_layers):
            if i != 0:
                prior_layer_outputs = [nr.out for nr in self.output_layer.neurons]
            for neuron in layer.neurons:
                neuron.net = neuron.bias + np.sum(np.multiply(
                    prior_layer_outputs, neuron.weights))

                neuron.out = neuron.activation_function(neuron.net)

    def backpropagate(self):
        pass
