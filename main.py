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

    wih = np.array([[-1, 2],
                    [3, -4]])

    who = np.array([[2, -1]])

    simple_network = Network(simple_network_shape)
    # simple_network._set_weights(ones)

    simple_network.hidden_layers[0].input_weights_from_matrix(wih)
    simple_network.output_layer.input_weights_from_matrix(who)

    w = simple_network.get_weights()
    print('weights:', w)

    print(simple_network.input_layer)
    print(simple_network.hidden_layers[0])
    print(simple_network.output_layer)

    input_vector = np.array([2, 3])
    simple_network.forwardpropagate(input_vector)
    out = simple_network.get_output()
    print('output: ', out)

    # TEST
    print('test...')
    y1 = np.matmul(wih, np.transpose(input_vector))
    vectorized_sigmoid = np.vectorize(sigmoid)
    y1_out = vectorized_sigmoid(y1)

    print(y1_out)

    y2 = np.matmul(who, np.transpose(y1_out))
    y2_out = vectorized_sigmoid(y2)

    print(y2_out)


if __name__ == '__main__':
    main()
