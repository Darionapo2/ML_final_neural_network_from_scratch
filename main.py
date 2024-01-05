from Network import Network
import numpy as np
from utils import sigmoid, ones


def read_dataset(path: str = 'datasets/digits/train_digits.dat'):
    with open(path, 'r+') as dataset:
        text = dataset.read()
        lines = text.split('\n')

        data = []
        del lines[-1]

        for line in lines:
            data.append(np.array(line.split(' '), dtype = int))

        # print(len(data[1]))


def main():
    digits_network_shape = [
        {'neurons_number': 256, 'activation_f': ''},
        {'neurons_number': 8, 'activation_f': 'sigmoid'},
        {'neurons_number': 8, 'activation_f': 'sigmoid'},
        {'neurons_number': 10, 'activation_f': 'sigmoid'}
    ]

    simple_network_shape = [
        {'neurons_number': 2, 'activation_f': ''},
        {'neurons_number': 2, 'activation_f': 'sigmoid'},
        {'neurons_number': 1, 'activation_f': 'sigmoid'}
    ]

    # digits_network = Network(digits_network_shape)

    wih = np.array([[1, 2],
                    [3, 4]])

    who = np.array([2, 1])

    simple_network = Network(simple_network_shape)
    # simple_network._set_weights(ones)

    weighted_layers = simple_network.get_weighted_layers()
    first_hidden_layer = weighted_layers[0]
    output_layer = weighted_layers[1]

    w = simple_network.get_weights()
    print(w)

    print(simple_network.input_layer)
    print(simple_network.hidden_layers[0])
    print(simple_network.output_layer)

    simple_network.forwardpropagate([1.2, 4.5])
    out = simple_network.get_output()
    print(out)

    print(sigmoid(1.2 + 4.5))
    print(sigmoid(1.2 + 4.5) * 2)
    print(sigmoid(sigmoid(1.2 + 4.5) * 2))


if __name__ == '__main__':
    main()
