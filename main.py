from Network import Network
import numpy as np


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

    digits_network = Network(digits_network_shape)


if __name__ == '__main__':
    main()
