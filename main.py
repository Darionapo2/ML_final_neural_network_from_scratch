import pprint

from Network import Network
import os
import numpy as np
from utils import sigmoid, ones, error, normal_distribution, derived_sigmoid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# TF_ENABLE_ONEDNN_OPTS=0
def read_dataset(path: str = 'datasets/digits/train_digits.dat') -> tuple:
    with open(path, 'r+') as dataset:
        lines = dataset.readlines()

        # Initialize empty lists to store data and expected_pred
        data = []
        expected_pred = []

        for line in lines[1:]:
            input_data = [int(num) for num in line.split()[:-10]]
            expected_output = [int(exp) for exp in line.split()[256:]]

            data.append(input_data)
            expected_pred.append(expected_output)


    return data, expected_pred

def read_dataset_dario(filename: str):
    X = []
    Y = []

    with open(filename, 'r+') as file:
        text = file.read()
        lines = text.split('\n')

        data_size, label_size, n_records = [int(v) for v in lines[0].split(' ')]

        for line in lines[1:]:
            if len(line) > 1:
                X.append([int(v) for v in line.split(' ')[:-label_size]])
                Y.append([int(v) for v in line.split(' ')[data_size:]])

    return X, Y


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


'''
def main2():
    digits_network_shape = [
        {'neurons_number': 2, 'activation_f': ''},
        {'neurons_number': 2, 'activation_f': 'sigmoid'},
        {'neurons_number': 1, 'activation_f': 'softmax'}
    ]

    wih = np.array([[1, 1, 2],
                    [2, 2, 1]])
    who = np.array([[1, 1, 1]])
    mu = []
    mu_bias = []
    for j, cha in enumerate(gradient_weights):
        mu.append([])
        for i, ch in enumerate(cha):
            mu[j].append(0.0)
        mu_bias.append(0.0)
    print(mu)

    best_val_loss = float('inf')
    tolerance = 0.2  # valore no sense
    input_v = [1, 0]

    d_network = Network(digits_network_shape)
    first_hidden_layer_weights = np.array(d_network.hidden_layers[0].get_weights())

    d_network.hidden_layers[0].input_weights_from_matrix(wih)
    d_network.output_layer.input_weights_from_matrix(who)

    d_network.forwardpropagate(input_v)
    out = d_network.get_output()
    print('output:', out)

    E = d_network.evaluate_results([0])
    print('E:', E)

    # to do: remember to insert the dataset as an attribute of the network class
    deltas = d_network.backpropagate([0], True)
    gradient_weights, gradient_bias = d_network.accumulatechange(deltas)

    mu, mu_bias = d_network.adjust_weights(gradient_weights, gradient_bias, 0.9, mu, mu_bias)

    # final_value = d_network.performance_evaluation(input_v,)
'''


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


def train():
    digits_network_shape = [
        {'neurons_number': 256, 'activation_f': ''},
        {'neurons_number': 6, 'activation_f': 'sigmoid'},
        {'neurons_number': 6, 'activation_f': 'sigmoid'},
        {'neurons_number': 10, 'activation_f': 'sigmoid'}
    ]

    input_vector, real_output = read_dataset()
    print(len(input_vector[0]))
    t_network = Network(digits_network_shape)
    t_network.set_weights()

    t_network.forwardpropagate(input_vector[0])
    print('output:', t_network.get_output())
    print('error between this output and the reference', t_network.evaluate_results(real_output))

    print('random weights: ', len(t_network.get_weights()[2]))
    print(len(real_output))
    deltas = t_network.backpropagate(real_output)
    gradient_w, gradient_b = t_network.accumulatechange(deltas)
    mu = []
    mu_bias = []
    for j, cha in enumerate(gradient_w):
        mu.append([])
        for i, ch in enumerate(cha):
            mu[j].append(0.0)
        mu_bias.append(0.0)

    mu, mu_bias = t_network.adjust_weights(gradient_w, gradient_b, 0.9, mu, mu_bias)

    print('random output:', t_network.get_output())

    t_network.forwardpropagate(input_vector)


def test3(weights: list, biases: list):
    simple_network_shape = [
        {'neurons_number': 4, 'activation_f': ''},
        {'neurons_number': 3, 'activation_f': 'sigmoid'},
        {'neurons_number': 2, 'activation_f': 'sigmoid'}
    ]

    X = [[1, 2, 3, 4], [2, 2, 2, 2]]
    Y = [[1, 1], [3, 3]]

    simple_network = Network(simple_network_shape)
    weights_tot_h_l = []
    for bias in biases[:-1]:
        weights_tot_h_l.append(bias)
    for weight in weights[0]:
        weights_tot_h_l.append(weight)

    hidden_layer_weights = np.transpose(weights_tot_h_l)

    weights_tot_o_l = []
    for bias in biases[1:]:
        weights_tot_o_l.append(bias)
    for weight in weights[1]:
        weights_tot_o_l.append(weight)
    output_layer_weights = np.transpose(weights_tot_o_l)
    print(hidden_layer_weights)
    print(output_layer_weights)

    simple_network.hidden_layers[0].input_weights_from_matrix(hidden_layer_weights)

    simple_network.output_layer.input_weights_from_matrix(output_layer_weights)
    simple_network.train(X, Y)

    print('pesi dopo back', simple_network.get_weights())

    # simple_network.forwardpropagate([2, 2, 1, 4])

    out = simple_network.get_output()
    print('output our model = ', out)


def test_keras():
    X = np.array(([[1, 2, 3, 4], [2, 2, 2, 2]]))
    Y = np.array(([[1, 1], [3, 3]]))

    model = Sequential()

    model.add(Input(shape = (4,)))
    model.add(Dense(units = 3, activation = 'sigmoid', name = 'hidden_layer_1'))
    model.add(Dense(units = 2, activation = 'sigmoid', name = 'output_layer'))
    model.summary()
    sgd = tf.keras.optimizers.SGD(learning_rate = 1, momentum = 1)
    model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

    l = [np.array([[1, 1, 2],
                   [1, 1, 2],
                   [2, 1, 2],
                   [1, 3, 2]]), np.array([0, 0, 0])]

    o = [np.array([[1, 1],
                   [1, 2],
                   [3, 1]]), np.array([0, 0])]
    model.layers[0].set_weights(l)
    print(model.layers[0].name)
    w = model.layers[0].get_weights()
    print(w)
    model.layers[1].set_weights(o)
    print(model.layers[1].name)
    w2 = model.layers[1].get_weights()

    print(w2)

    w = model.get_weights()
    weights = []
    biases = []

    for w1, b in zip(w[::2], w[1::2]):
        weights.append(w1)
        biases.append(b)
    # weights.append([w[i] for i in range(0, len(w), 2)])
    # biases.append([w[i] for i in range(1, len(w), 2)])

    model.fit(X, Y, epochs = 0, batch_size = 1)

    # model.predict(np.array(([[2, 2, 1, 4]])))

    result = model.predict(X)
    print('output keras = ', result)

    return weights, biases


def test4():
    digits_network_shape = [
        {'neurons_number': 256, 'activation_f': ''},
        {'neurons_number': 6, 'activation_f': 'sigmoid'},
        {'neurons_number': 6, 'activation_f': 'sigmoid'},
        {'neurons_number': 10, 'activation_f': 'sigmoid'}
    ]

    X, y = read_dataset_dario('datasets/digits/train_digits.dat')
    X_test, y_test = read_dataset_dario('datasets/digits/test_digits.dat')

    d_network = Network(digits_network_shape)
    d_network.set_weights()

    nepochs = 10000
    for i in range(nepochs):
        d_network.train(X, y)

    count = 0
    for X_t, y_t in zip(X_test, y_test):
        d_network.forwardpropagate(X_t)
        out = d_network.get_output()
        if out.index(max(out)) == y_test.index(1):
            count += 1
        print('out', out)

    acc = count/len(X_test)
    print('acc:', acc)


def xor_train():
    xor_network = [
        {'neurons_number': 2, 'activation_f': ''},
        {'neurons_number': 2, 'activation_f': 'sigmoid'},
        {'neurons_number': 2, 'activation_f': 'sigmoid'}
    ]

    xor_network = Network(xor_network)
    xor_network.set_weights()
    print(xor_network.get_weights())

    X, Y = read_dataset_dario('datasets/xor/train_xor.dat')
    print('X:', X)

    nepochs = 10000
    # diff = []
    # w = []
    for i in range(nepochs):
        loss = xor_network.train(X, Y)
        print(xor_network.get_weights())


    xor_network.forwardpropagate([-1,-1])
    out00 = xor_network.get_output()
    xor_network.forwardpropagate([-1,1])
    out01 = xor_network.get_output()
    xor_network.forwardpropagate([1,-1])
    out10 = xor_network.get_output()
    xor_network.forwardpropagate([1,1])
    out11 = xor_network.get_output()

    print('out:', out00, out01, out10, out11)

def test_keras2():

    X, Y = read_dataset_dario('datasets/xor/train_xor.dat')
    print(X)

    model = Sequential()

    model.add(Input(shape=(2,)))
    model.add(Dense(units=2, activation='tanh', name='hidden_layer_1'))
    model.add(Dense(units=2, activation='sigmoid', name='output_layer'))
    model.summary()

    model.compile(loss = 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X, Y, epochs = 10000, batch_size=1)

    X_val, Y_val = read_dataset_dario('datasets/xor/test_xor.dat')

    result = model.predict([[1,1]])
    print(result)

def test_one_neuron():
    one_network = [
        {'neurons_number': 1, 'activation_f': ''},
        {'neurons_number': 1, 'activation_f': 'sigmoid'},
    ]

    one_network = Network(one_network)
    one_network.set_weights()

    for i in range(10000):
        one_network.train([[1]], [[1]])

        print('w ',one_network.get_weights())

    for i in range(1000):
        one_network.forwardpropagate([[1]])
        print('out:', one_network.get_output())


if __name__ == '__main__':
    # main2()
    test4()
    # w, b = test_keras()
    # test3(w, b)
    # xor_train()
    # test_keras2()
    #test_one_neuron()
