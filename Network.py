from Neuron import Neuron
from Layer import Layer
from utils import sigmoid
from utils import activation_functions


class Network:
    input_layer: Layer
    output_layer: Layer
    hidden_layers: list[Layer]

    def __init__(self, network_shape: list[dict]):
        input_layer_shape = network_shape[0]
        k = input_layer_shape['neurons_number']
        self.input_layer = Layer(k, activation_functions['identity'])

        for hidden_layer_shape, i in zip(network_shape[1:-1], range(1, len(network_shape) + 1)):
            n = hidden_layer_shape['neurons_number']
            selected_hl_act_f = hidden_layer_shape['activation_function']
            if selected_hl_act_f in activation_functions:
                hl_act_f = activation_functions[selected_hl_act_f]
            else:
                print(f'Error: Layer n. {i} activation function not avaiable. '
                      'Sigmoid is used instead.')
                hl_act_f = sigmoid

            if i == len(network_shape) + 1:
                self.output_layer = Layer(n, hl_act_f)
            else:
                self.hidden_layers.append(Layer(n, hl_act_f))

    def backpropagate(self):
        pass

    def forwardpropagate(self):
        pass
