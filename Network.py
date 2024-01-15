from Neuron import Neuron
from Layer import Layer
from utils import sigmoid, activation_functions, normal_distribution, distance
import numpy as np
from typing import Callable


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

    def get_hidden_layers(self):
        return self.hidden_layers

    def get_input_layer(self):
        return self.input_layer

    def get_output_layer(self):
        return self.output_layer

    def get_layers(self):
        return [self.input_layer] + self.hidden_layers + [self.output_layer]

    def get_weighted_layers(self):
        return self.hidden_layers + [self.output_layer]

    def set_weights(self):
        self._set_weights(criteria = normal_distribution)

    def _set_weights(self, criteria: Callable):
        n_prev_layer = 0

        for layer in self.get_weighted_layers():
            for neuron in layer.neurons:
                if n_prev_layer == 0:
                    n_prev_layer = self.input_layer.n
                neuron.weights = criteria(n_prev_layer)
            n_prev_layer = layer.n

    def forwardpropagate(self, input_data: np.array):

        self.input_layer.feed(input_data)
        self.input_layer.activate()

        all_layers = self.get_layers()

        for prior_layer, current_layer in zip(all_layers, all_layers[1:]):
            prior_layer_outputs = prior_layer.get_output_values()

            current_layer_input = [
                nr.bias + np.sum(
                    np.multiply(prior_layer_outputs, nr.weights)
                ) for nr in current_layer.neurons
            ]

            current_layer.feed(current_layer_input)
            current_layer.activate()

    def evaluate_results(self, reference: list[float]):
        results = np.ndarray(self.output_layer.get_output_values())
        return distance(results, np.ndarray(reference))

    def get_output(self):
        return self.output_layer.get_output_values()

    def get_weights(self):
        weights = [hl.get_weights() for hl in self.hidden_layers]
        weights.append(self.output_layer.get_weights())
        return weights

    def backpropagate(self):
        pass
