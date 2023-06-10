# This is a sample Python script.
import datetime
import itertools
import os
import time

import keras.metrics
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers.optimizer_v2.adam import Adam
from keras.utils import plot_model
from keras.backend import clear_session
from pandas import DataFrame

from sklearn.model_selection import train_test_split

from numpy.random import default_rng
import bisect

from data.data_crate import create_data

rng = default_rng(seed=42)


# Yuval
# todo - add dipper (2 layers) networks and more layers.
# todo - try 3,5,7 and 3,5,11 and 2,5,7 (multiply of those number).
# todo - learn until near saturation and than swap.

# TODO - to check if the NN memorize the data or not, i.e check if it can.
# TODO - make sure all the data is unique.
# TODO - start from one divider and than add more.
# TODO - check what is minimal layers needed to succeed in validation.
# TODO - check how that history is saved.
# TODO - girvan newman algorithm


def convert_to_graph(model):
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


def model_crate(num_inputs, layers):
    model = Sequential()
    if layers == 0:
        model.add(Dense(1, input_dim=num_inputs, activation='sigmoid'))
    else:
        model.add(Dense(num_inputs, input_dim=num_inputs, activation='relu'))
        for i in range(layers - 1):
            model.add(Dense(num_inputs, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[keras.metrics.BinaryAccuracy()])
    plot_model(model, to_file=f'model_{layers}_layers.png', show_shapes=True, show_layer_names=True)
    return model


def MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model: Sequential, layers, total_epochs_for_each_label=300, round_epoch=10):
    cnt = 0
    accuracy_all_train = []
    accuracy_all_test = []
    accuracy_train = {}
    accuracy_test = {}
    for divider in numbers_to_divide_by:
        accuracy_train[divider] = []
        accuracy_test[divider] = []
    while cnt < total_epochs_for_each_label:
        for j, number in enumerate(numbers_to_divide_by):
            cur_train = y_train.iloc[:, j]
            cur_test = y_test.iloc[:, j]
            model.fit(X_train, cur_train, validation_data=[X_test, cur_test], epochs=round_epoch, verbose=0, shuffle=True,
                      use_multiprocessing=True, workers=4)
            accuracy_all_train.extend(model.history.history['binary_accuracy'])
            accuracy_all_test.extend(model.history.history['val_binary_accuracy'])
            accuracy_train[number].extend(model.history.history['binary_accuracy'])
            accuracy_test[number].extend(model.history.history['val_binary_accuracy'])
        cnt += round_epoch
        print(f'epoch {cnt} out of {total_epochs_for_each_label}')
    plt.plot(accuracy_all_train)
    plt.title(
        f'MVG accuracy train, layers {layers}, round epoch {round_epoch}  {numbers_to_divide_by}')
    plt.show()
    plt.plot(accuracy_all_test)
    plt.title(
        f'MVG accuracy test, layers {layers}, round epoch {round_epoch}  {numbers_to_divide_by}')
    plt.show()
    for divider in numbers_to_divide_by:
        plt.plot(accuracy_train[divider], label=f'train {divider}')
        plt.plot(accuracy_test[divider], label=f'test {divider}')
        plt.title(f'MVG, layers {layers}, round epoch {round_epoch}, divider {divider}')
        plt.legend()
        # save fig in folder named by date
        plt.savefig(f'MVG, layers {layers}, round epoch {round_epoch}, divider {divider}.png')
        plt.show()
    # run the algorithm on the model
    # check if the algorithm found the correct number of dividers
    # check if the algorithm found the correct dividers


def get_all_premonition(numbers, amount_of_elements):
    combinations = list(itertools.combinations(numbers, amount_of_elements))
    for i in range(len(combinations)):
        combinations[i] = np.prod(combinations[i])
    return combinations


def FG(X_train, y_train, X_test, y_test, divider, model: Sequential, layers, iterations=100):
    model.fit(X_train, y_train, epochs=iterations, validation_data=(X_test, y_test), shuffle=True, use_multiprocessing=True, workers=4)
    # find the point where the accuracy is 0.9
    accuracy_all_train = model.history.history['binary_accuracy']
    accuracy_all_test = model.history.history['val_binary_accuracy']
    point_where_accuracy_is_09 = -1
    if accuracy_all_test[-1] > 0.9:
        point_where_accuracy_is_09 = bisect.bisect_left(accuracy_all_test, 0.9)
        print(f'point where accuracy is 0.9 is {point_where_accuracy_is_09}')
    plt.plot(accuracy_all_train)
    plt.plot(accuracy_all_test)
    if point_where_accuracy_is_09 != -1:
        plt.plot(point_where_accuracy_is_09, 0.9, 'ro')
        plt.text(point_where_accuracy_is_09, 0.9, f'({point_where_accuracy_is_09}, 0.9)')
    plt.title(f'FG, Divider {divider}, layers {layers}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.savefig(f'FG, Divider {divider}, layers {layers}.png')
    plt.show()


from ann_visualizer.visualize import ann_viz


def model_to_graph(model: Sequential):
    dot_img_file = 'model_1.png'
    ann_viz(model, title="My first neural network")
    G = convert_to_graph(model)
    print(G.nodes)
    print(G.edges)
    nx.draw(G)
    plt.show()


def main():
    num_inputs = 32
    sample_size = 20000
    numbers_to_divide_by = get_all_premonition([3, 5, 7, 9], amount_of_elements=3)
    max_number = 100000
    # layers 3 - 10
    # switch rate 2 - 20
    # modularity
    # start from one number
    df = create_data(num_inputs, sample_size, max_number, numbers_to_divide_by, use_old_data=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print some statistics about the data
    print(df['decimal'].describe())
    print(f"amount of unique values: {df['decimal'].nunique()}, total amount of values: {len(df['decimal'])}")
    X = df.filter(regex='input')
    y = df.filter(regex='divide_by')
    # find thw is the max number of bits in the input
    # split the data into training and testing
    y_test: DataFrame
    y_train: DataFrame
    X_test: DataFrame
    X_train: DataFrame
    X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
    # move to date folder
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    # time_start = time.time()
    for layers in range(6, 7):
        clear_session()
        model = model_crate(num_inputs, layers)
        FG(X_train, y_train.iloc[:, -1], X_test, y_test.iloc[:, -1], numbers_to_divide_by[-1], model, layers, iterations=20)
        model_to_graph(model)
        del model
    # time_end = time.time()
    # print(f'time to run FG: {time_end - time_start}')
    # time_start = time.time()

    # for layers in range(2, 3):
    #     clear_session()
    #     model = model_crate(num_inputs, layers)
    #     MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, layers, total_epochs_for_each_label=300, round_epoch=20)
    #     del model
    # time_end = time.time()
    # print(f'time to run MVG: {time_end - time_start}')
    # MVG(X_train, y_train, X_test, y_test, model, num_epochs, numbers_to_divide_by)


if __name__ == '__main__':
    main()

    # FG(X_train, y_train.iloc[:, 0], X_test, y_test.iloc[:, 0], numbers_to_divide_by[0], model)
    # MVG(X_train, y_train, X_test, y_test, model, num_epochs, numbers_to_divide_by)

    # learn_on_one_divider(X_train, y_train, X_test, y_test, model, num_epochs, numbers_to_divide_by)
