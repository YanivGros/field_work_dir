# Description: Utility functions for the project
import argparse
import itertools
import os

import networkx as nx
import numpy as np

# keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam
from keras.metrics import BinaryAccuracy


def crate_and_cd_to_dir(dir_name: str) -> None:
    """
    Crate a directory and change the current working directory to it
    :param dir_name: the name of the directory
    :return: None
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)


def get_all_premonition_mult(numbers: list, amount_of_elements: int) -> list:
    """
    Get all possible combinations of numbers with length of amount_of_elements
    :param numbers: the numbers to combine
    :param amount_of_elements: the length of the combinations
    :return: a list of all possible combinations Multiplied
    """
    ret_list = []
    combinations = list(itertools.combinations(numbers, amount_of_elements))
    for i in range(len(combinations)):
        ret_list.append(np.prod(combinations[i]))
    return ret_list


def crate_parser():
    """
    Crate a parser for the command line arguments
    :return: the arguments
    """
    parser = argparse.ArgumentParser(prog="MVG")
    parser.add_argument('-n', '--num_inputs', type=int, default=32)
    parser.add_argument('-d', '--data_size', type=int, default=20_000)
    parser.add_argument('-m', '--max_number', type=int, default=100_000)
    parser.add_argument('-l', '--layers', type=int, default=5)
    parser.add_argument('--learning_type', type=str, choices=['FG', 'MVG'], default='MVG')
    parser.add_argument('-i', '--iterations_fg', type=int, default=100)
    parser.add_argument('-r', '--iterations_each_round_mvg', type=int, default=40)
    parser.add_argument('-e', '--amount_of_rounds_mvg', type=int, default=20)
    return parser.parse_args()

from keras.regularizers import l1
def model_crate(num_inputs, layers, width=1):
    """
    Crate a model with the given parameters
    :param width: the width of the model
    :param num_inputs: number of bits that the model will get
    :param layers: number of layers in the model
    :return: sequential NN model
    """
    model = Sequential()
    should_use_kernel_regularize = False
    if layers == 0:
        model.add(Dense(1, input_dim=num_inputs, activation='sigmoid'))
    else:
        model.add(Dense(num_inputs * width, input_dim=num_inputs, activation='relu', kernel_regularizer=l1(0.01)))
        for i in range(layers - 1):
            model.add(Dense(num_inputs * width, activation='relu', kernel_regularizer=l1(0.01)))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[BinaryAccuracy()])
    print(model.summary())
    return model


def convert_to_graph(model: Sequential):
    """
    Convert a model to a NX graph
    :param model: sequential keras model
    :return: nx graph
    """
    G = nx.Graph()
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            # Get the weights and biases from the layer
            W, b = weights[0], weights[1]
            num_nodes_prev, num_nodes = W.shape

            # Add nodes to the graph
            for i in range(num_nodes):
                G.add_node((layer.name, i))

            # Add edges with weights to the graph
            for i in range(num_nodes_prev):
                for j in range(num_nodes):
                    weight = W[i, j]
                    G.add_edge((layer.name, i), (layer.name, j), weight=weight)
    return G


def print_with_separator(string: str):
    """
    Print a string with a separator
    :param string: the string to print
    :return: None
    """
    print(f"{'-' * 20}\n{string}\n{'-' * 20}")
