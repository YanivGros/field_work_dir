# Description: Utility functions for the project
import argparse
import itertools
import os

import networkx as nx
import numpy as np
from keras.layers import Dense
from keras.metrics import BinaryAccuracy
# keras imports
from keras.models import Sequential


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
    parser.add_argument('-m', '--max_number', type=int, default=2_000_000)
    parser.add_argument('-l', '--layers', type=int, default=4)
    # parser.add_argument('-l', '--layers', type=int, default=2)
    parser.add_argument('--learning_type', type=str, choices=['FG', 'MVG'], default='FG')
    parser.add_argument('-i', '--iterations_fg', type=int, default=10_000)
    parser.add_argument('-r', '--iterations_each_round_mvg', type=int, default=15)
    parser.add_argument('-e', '--amount_of_rounds_mvg', type=int, default=61)
    # parser.add_argument('-w', '--midth_multiplayer', type=int, default=1)
    parser.add_argument('-w', '--width_multiplayer', type=int, default=3)
    return parser.parse_args()


def model_crate(num_inputs, layers, width_multiplayer=1, regularizer=None):
    """
    Crate a model with the given parameters
    :param regularizer: the regularizer to use
    :param width_multiplayer: the width of the model
    :param num_inputs: number of bits that the model will get
    :param layers: number of layers in the model
    :return: sequential NN model
    """
    model = Sequential()
    if layers == 0:
        model.add(Dense(1, input_dim=num_inputs, activation='sigmoid'))
    else:
        model.add(Dense(num_inputs * width_multiplayer, input_dim=num_inputs, activation='relu', kernel_regularizer=regularizer))
        for i in range(layers - 1):
            model.add(Dense(num_inputs * width_multiplayer, activation='relu', kernel_regularizer=regularizer))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=[BinaryAccuracy()])
    return model


def convert_to_graph(model):
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

