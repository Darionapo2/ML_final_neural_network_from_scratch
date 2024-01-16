from Network import Network
import numpy as np
from utils import sigmoid, ones, error, normal_distribution


def read_dataset(path: str = 'datasets/digits/train_digits.dat') -> list:
    with open(path, 'r+') as dataset:
        text = dataset.read()
        lines = text.split('\n')

        data = []
        del lines[-1]

        for line in lines:
            data.append(np.array(line.split(' '), dtype = int))

    return data


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

    # TEST WITH DIGITS DATASET:
    data = read_dataset('datasets/digits/test_digits.dat')
    input_data = data[1][:-10]
    reference = data[1][256:]
    print(len(input_data), len(reference))

    digits_network = Network(digits_network_shape)
    digits_network.forwardpropagate(input_data)
    output = digits_network.output_layer.get_output_values()

    # digits_network.backpropagate(reference)

    print('first hidden layer weights:', digits_network.hidden_layers[0].get_weights())


def main2():
    digits_network_shape = [
        {'neurons_number': 2, 'activation_f': ''},
        {'neurons_number': 2, 'activation_f': 'identity'},
        {'neurons_number': 1, 'activation_f': 'identity'}
    ]

    wih = np.array([[1, 2],
                    [3, 4]])

    who = np.array([[2, 3]])

    d_network = Network(digits_network_shape)
    first_hidden_layer_weights = np.array(d_network.hidden_layers[0].get_weights())
    all_network_weights = d_network.get_weights()
    print(all_network_weights)

    d_network.hidden_layers[0].input_weights_from_matrix(wih)
    d_network.output_layer.input_weights_from_matrix(who)

    d_network.forwardpropagate([1, 3])
    out = d_network.get_output()
    print('output:', out)

    E = d_network.evaluate_results([1])
    print('E:', E)

    d_network.backpropagate([1])


if __name__ == '__main__':
    main2()
