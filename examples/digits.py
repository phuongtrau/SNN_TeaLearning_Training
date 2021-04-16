# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
import random

import numpy as np
from sklearn import datasets
from network import SpikingNetwork

from mlxtend.data import loadlocal_mnist
import platform


MAX_DIGIT = 10
MINI_BATCH_SIZE = 5000

# xem lai cach xet threshold de convert sang spike signals
def convert_image(image): # flatten out into a vector
    return np.array([[x > 0 for j, x in enumerate(y)] for i, y in enumerate(image)]).reshape((784, 1))


def make_number(number): # number from 0 to 9
    zeros = np.zeros((MAX_DIGIT, 1))
    zeros[number] = 1
    return zeros


def main():
    if '--help' in sys.argv[1:]:
        print('usage: python {} [--load (filename)] [--no-save] [--draw]'.format(sys.argv[0]))
        return

    # initialize a network
    network = SpikingNetwork(draw_spike='--draw' in sys.argv[1:])
    network.add(784)
    network.add(300, -0.5)
    network.add(MAX_DIGIT, -0.3)

    # preparing data
    train_data = []
    test_data = []

    print('MNIST Loading...', end='')
    # mnist = datasets.fetch_openml('mnist-original')
    # mnist = datasets.fetch_openml('mnist-784')
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
                images_path='train-images-idx3-ubyte', 
                labels_path='train-labels-idx1-ubyte')
        X_test, y_test = loadlocal_mnist(
                images_path='t10k-images-idx3-ubyte', 
                labels_path='t10k-labels-idx1-ubyte')
    else:
        X, y = loadlocal_mnist(
                images_path='train-images.idx3-ubyte', 
                labels_path='train-labels.idx1-ubyte')
        X_test, y_test = loadlocal_mnist(
                images_path='t10k-images.idx3-ubyte', 
                labels_path='t10k-labels.idx1-ubyte')

    print('OK')

    for number in range(MAX_DIGIT):
        # test and train data
        # both_data = [{
        #     'x': convert_image(data.reshape((28, 28))),
        #     'y': make_number(number) # one-hot vector
        # } for data in mnist['data'][mnist['target'] == number]]
        
        both_data = [{
            'x': convert_image(data.reshape((28, 28))),
            'y': make_number(number)
        } for data in X[y == number]]
        both_data_tmp = [{
            'x': convert_image(data.reshape((28, 28))),
            'y': make_number(number)
        } for data in X_test[y_test == number]]
        
        both_data.extend(both_data_tmp)

        #train_data.extend(both_data[:100])
        #test_data.extend(both_data[100:105])
        #train_data.extend(both_data[:1])
        #test_data.extend(both_data[:1])
        
        test_data.extend(both_data[:10])
        train_data.extend(both_data[10:])

    # check loading model
    if '--load' in sys.argv[1:]:
        print('Loading...', end='')
        load_index = sys.argv.index('--load')
        if len(sys.argv) > load_index + 1 and os.path.exists(sys.argv[load_index + 1]):
            network.load(path=sys.argv[load_index + 1])
        else:
            network.load()
        print('OK')

    for i in range(10000000):
        network.forward(np.concatenate([data['x'] for data in test_data], axis=1), 50)
        answer = np.concatenate([data['y'] for data in test_data], axis=1)
        infer = network.infer(display_no_spike=i % 10 == 0)
        complete = False
        if np.all(np.absolute(infer - answer) < 0.1):
            complete = True
        if i % 5 == 0 or complete:
            print('In {}'.format(i))
            print('answer:\n{}'.format(answer))
            print('infer:\n{}'.format(infer))
        if complete:
            print('Complete!')
            if '--no-save' not in sys.argv[1:]:
                print('Saving...', end='')
                network.save()
                print('OK')
            return

        random.shuffle(train_data)
        for j in range(0, len(train_data), MINI_BATCH_SIZE):
            network.forward(np.concatenate([data['x'] for data in train_data[j:j + MINI_BATCH_SIZE]], axis=1), 50)
            network.backward(np.concatenate([data['y'] for data in train_data[j:j + MINI_BATCH_SIZE]], axis=1))


if __name__ == '__main__':
    main()
