from Network import Network
import numpy as np
from utils import sigmoid, ones, error, normal_distribution, derived_sigmoid


def read_dataset(path: str = 'datasets/digits/train_digits.dat') -> list:
    with open(path, 'r+') as dataset:
        text = dataset.read()
        lines = text.split('\n')

        data = []
        del lines[-1]

        for line in lines:
            data.append(np.array(line.split(' '), dtype=int))

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
        {'neurons_number': 2, 'activation_f': 'sigmoid'},
        {'neurons_number': 1, 'activation_f': 'sigmoid'}
    ]

    wih = np.array([[1, 1, 2],
                    [2, 2, 1]])

    who = np.array([[1, 1, 1]])

    d_network = Network(digits_network_shape)
    first_hidden_layer_weights = np.array(d_network.hidden_layers[0].get_weights())

    d_network.hidden_layers[0].input_weights_from_matrix(wih)
    d_network.output_layer.input_weights_from_matrix(who)

    input_v = [1, 0]

    d_network.forwardpropagate(input_v)
    out = d_network.get_output()
    print('output:', out)

    E = d_network.evaluate_results([0])
    print('E:', E)

    # to do: remember to insert the dataset as an attribute of the network class
    deltas = d_network.backpropagate([0], True)
    gradient_weights, gradient_bias = d_network.accumulatechange(deltas)
    '''
    for i, cha in enumerate(change):
        for j, ch in enumerate(cha):
            print(f'change[{i}][{j}]: {ch}')
    print('bias: ', bias)

    for nr in d_network.input_layer.neurons:
        print(nr)
    '''
    learning_rate = 0.1
    d_network.adjust_weights(gradient_weights, gradient_bias, learning_rate)












def test():

    wih = np.array([[1, 1, 2],
                    [2, 2, 1]])

    who = np.array([[1, 1, 1]])

    print('TEST---------------------')
    input_v = [1, 1, 0]
    res_1_l = np.matmul(wih, np.transpose(input_v))
    print('propagazione dell input al primo hidden layer:', res_1_l)
    act_res = [sigmoid(r) for r in res_1_l]
    print('sigmoid applicata ai net value ottenuti:', act_res)

    print('secondo layer (output)--------------')

    res_2_l = np.matmul(who, np.transpose([1.0] + act_res))
    out = [sigmoid(r) for r in res_2_l]
    print('out', out)

    reference = 0

    E = (reference - out[0]) ** 2
    print('error: ', E)

    delta1 = (-2) * (reference - out[0]) * derived_sigmoid(sum(res_2_l))
    print('delta1: ', delta1)




if __name__ == '__main__':
    main2()
    # test()
