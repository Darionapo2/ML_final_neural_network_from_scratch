import pprint

from Layer import *
from utils import sigmoid, activation_functions, derivatives, error, get_difference, \
    normal_distribution
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
        self._set_weights(criteria=normal_distribution)

    def _set_weights(self, criteria: Callable):
        n_prev_layer = 0

        for layer in self.get_weighted_layers():
            for neuron in layer.neurons:
                if n_prev_layer == 0:
                    n_prev_layer = self.input_layer.n
                neuron.weights = criteria(n_prev_layer)
                neuron.bias = criteria(1)[0]
            n_prev_layer = layer.n

    def forwardpropagate(self, input_data: np.array):

        self.input_layer.feed(input_data)
        self.input_layer.activate()

        all_layers = self.get_layers()
        for previous_layer, current_layer in zip(all_layers, all_layers[1:]):
            previous_layer_outputs = previous_layer.get_output_values()

            current_layer_input = [
                nr.bias + np.sum(
                    np.multiply(previous_layer_outputs, nr.weights)
                ) for nr in current_layer.neurons
            ]

            current_layer.feed(current_layer_input)
            current_layer.activate()

    def evaluate_results(self, reference: list[float]):
        results = np.array(self.output_layer.get_output_values())
        return error(results, np.array(reference))

    def get_output(self):
        return self.output_layer.get_output_values()

    def get_weights(self):
        weights = [hl.get_weights() for hl in self.hidden_layers]
        weights.append(self.output_layer.get_weights())

        return weights

    def backpropagate(self, d: list[float], derivative_const: bool = False) -> list[np.array]:
        delta = []

        outs = self.output_layer.get_output_values()
        diff_vector = get_difference(d, outs)

        out_activation_f = self.output_layer.activation_f.__name__
        derived_act_function = derivatives[out_activation_f]

        derivs = [derived_act_function(nr.net) for nr in self.output_layer.neurons]
        print('net', [nr.net for nr in self.output_layer.neurons])
        print('derivs:', derivs)
        if derivative_const:
            diff_vector = np.multiply(diff_vector, -2)
        delta.append(np.multiply(diff_vector, derivs))
        reversed_layers = self.get_weighted_layers()[::-1]
        # previous_layer ==> h + 1, in respect of the inverted order of layers
        # current_layer ==> h
        previous_layer_index = 0  # to keep count of which layer number is the previous layer
        for previous_layer, current_layer in zip(reversed_layers, reversed_layers[1:]):
            new_delta = []
            for j, current_neuron in enumerate(current_layer.neurons):
                inc_weighted_error = 0
                for i, prev_layer_nr in enumerate(previous_layer.neurons):
                    inc_weighted_error += prev_layer_nr.weights[j] * delta[previous_layer_index][i]
                    # print('incremental weighted error:', inc_weighted_error)

                current_layer_act_f = current_layer.activation_f.__name__
                current_layer_derived_act_f = derivatives[current_layer_act_f]
                # print('derivative(net):', current_layer_derived_act_f(current_neuron.net))

                new_delta.append(inc_weighted_error * current_layer_derived_act_f(
                    current_neuron.net))

            delta.append(np.array(new_delta))
            previous_layer_index += 1

        # print('delta:', delta[::-1])
        return delta[::-1]

    # secondo me è da aggiungere anche la modifica del delta passo per passo => delta=delta + change
    def accumulatechange(self, deltas: list[np.array]) -> tuple[list[np.array], list]:
        all_layers = self.get_layers()
        change = []
        curr_layer_index = 0
        bias = []
        for previous_layer, current_layer in zip(all_layers, all_layers[1:]):
            for j, current_neuron in enumerate(current_layer.neurons):
                new_change = []
                for i, prev_layer_nr in enumerate(previous_layer.neurons):
                    # print(' previous_layer.neurons[i].out ', i, previous_layer.neurons[i].out)
                    new_change.append(deltas[curr_layer_index][j] * previous_layer.neurons[i].out)
                bias.append(deltas[curr_layer_index][j])
                change.append(new_change)
            curr_layer_index += 1

        return change, bias

    # posso palesemente usare direttamente current neuron weights ma non mi sembra bello bho, comunque lo toglieremo
    def adjust_weights(self, weights_gradient: list[np.array], bias_gradient: list, learning_rate: float,
                       mu: list[np.array], mu_bias: np.array) -> tuple[list[list], list]:
        layers = self.get_weighted_layers()
        new_mu = []
        new_mu_bias = []
        for previous_layer, current_layer in zip(layers[:-1], layers[1:]):
            for j, current_neuron in enumerate(current_layer.neurons):
                updated_weights = [wh for wh in current_neuron.weights]
                g_weights_layer = weights_gradient[j]
                mu_layer = []
                for i, previous_neuron in enumerate(previous_layer.neurons):
                    mu_layer.append(learning_rate * g_weights_layer[i])
                    updated_weights[i] -= (learning_rate * g_weights_layer[i]) + mu[j][i]
                current_neuron.weights = updated_weights
                current_neuron.bias -= learning_rate * bias_gradient[j] + mu_bias[j]
                new_mu.append(mu_layer)
                new_mu_bias.append(learning_rate * bias_gradient[j])

        return new_mu, new_mu_bias
