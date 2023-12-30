from Neuron import Neuron
from Layer import Layer
from utils import sigmoid
from utils import activation_functions


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

    def __str__(self):
        str_hls = [str(hl) for hl in self.hidden_layers]
        return f'<Network({str(self.input_layer)}, {", ".join(str_hls)}, {str(self.output_layer)})>'

    def backpropagate(self):
        pass

    def forwardpropagate(self):
        pass
