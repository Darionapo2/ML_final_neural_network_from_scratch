from Neuron import Neuron
class Network:
    input_neurons = []


    network_shape = {
        'layers': {
            'input': {
                'k': 256,
                'activation_function': 'sigmoid'
            },

            'hidden1': {
                'n1': 10,
                'activation_function': 'sigmoid'
            },

            'hidden2': {
                'n2': 10,
                'activation_function': 'sigmoid'
            },

            # ...

            'output': {
                'J': 10,
                'activation_function': 'softmax'
            }
        }
    }

    def __init__(self, shape: dict):
        pass


    def backpropagate(self):
        pass

    def forwardpropagate(self):
        pass