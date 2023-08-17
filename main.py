# This is a sample Python script.
import argparse
import datetime
import itertools
import os
import threading
import time
from timeit import timeit

import keras.metrics
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import *
from ann_visualizer.visualize import ann_viz
from networkx.algorithms.community import girvan_newman

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers.legacy import Adam
from keras.utils import plot_model
from keras.backend import clear_session
from pandas import DataFrame

from sklearn.model_selection import train_test_split
import community

from numpy.random import default_rng
import bisect

import tensorflow as tf

from data.data_crate import create_data

rng = default_rng(seed=42)


def MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model: Sequential, layers, num_round=10, epoch_per_round=10, path="./"):
    accuracy_columns = ['round', 'epoch', 'divider', 'train_accuracy', 'test_accuracy']
    accuracy_df = pd.DataFrame(columns=accuracy_columns)

    for i in range(num_round):
        for j, number in enumerate(numbers_to_divide_by):
            cur_train = y_train.iloc[:, j]
            cur_test = y_test.iloc[:, j]
            model.fit(X_train, cur_train, validation_data=[X_test, cur_test], epochs=epoch_per_round, verbose=1,
                      shuffle=True,
                      use_multiprocessing=True, workers=16)

            for epoch, (train_acc, test_acc) in enumerate(
                    zip(model.history.history['binary_accuracy'], model.history.history['val_binary_accuracy'])):
                accuracy_df.loc[len(accuracy_df)] = [i, epoch, number, train_acc, test_acc]
        print(f'round {i} out of {num_round}')
        if i % 5 == 0:
            plt.plot(accuracy_df['train_accuracy'])
            plt.plot(accuracy_df['test_accuracy'])
            plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, divider {numbers_to_divide_by}')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(path, f'MVG, layers {layers}, round epoch {epoch_per_round} round {i}.png'))
            plt.show()
            for divider in numbers_to_divide_by:
                divider_data = accuracy_df[accuracy_df['divider'] == divider]
                plt.plot(divider_data['train_accuracy'].reset_index(drop=True), label=f'train {divider}')
            plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, {numbers_to_divide_by}, separate graphs')
            plt.legend()
            plt.savefig(
                os.path.join(path,
                             f'MVG, layers {layers}, round epoch {epoch_per_round}, {numbers_to_divide_by}, round {i} separate graphs.png'))
            plt.show()
            accuracy_df.to_csv(os.path.join(path, f'MVG, layers {layers}, round epoch {epoch_per_round}.csv'), index=False)
    plt.plot(accuracy_df['train_accuracy'])
    plt.plot(accuracy_df['test_accuracy'])
    plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, divider {numbers_to_divide_by}')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, f'MVG, layers {layers}, round epoch {epoch_per_round}.png'))
    plt.show()
    accuracy_df.to_csv(os.path.join(path, f'MVG, layers {layers}, round epoch {epoch_per_round}.csv'), index=False)

    for divider in numbers_to_divide_by:
        divider_data = accuracy_df[accuracy_df['divider'] == divider]
        plt.plot(divider_data['train_accuracy'].reset_index(drop=True), label=f'train {divider}')
    plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, {numbers_to_divide_by}, separate graphs')
    plt.legend()
    plt.savefig(os.path.join(path, f'MVG, layers {layers}, round epoch {epoch_per_round}, {numbers_to_divide_by}, separate graphs.png'))
    plt.show()


def FG(X_train, y_train, X_test, y_test, divider, model: Sequential, layers, iterations=100, path="./"):
    model.fit(X_train, y_train, epochs=iterations, validation_data=(X_test, y_test), shuffle=True, use_multiprocessing=True, workers=8, )
    accuracy_all_train = model.history.history['binary_accuracy']
    accuracy_all_test = model.history.history['val_binary_accuracy']
    if accuracy_all_test[-1] > 0.9:
        point_where_accuracy_is_09 = bisect.bisect_left(accuracy_all_test, 0.9)
        print(f'point where accuracy is 0.9 is {point_where_accuracy_is_09}')
        plt.plot(point_where_accuracy_is_09, 0.9, 'ro')
        plt.text(point_where_accuracy_is_09, 0.9, f'({point_where_accuracy_is_09}, 0.9)')
    plt.plot(accuracy_all_train)
    plt.plot(accuracy_all_test)
    plt.title(f'FG, Divider {divider}, layers {layers}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    if accuracy_all_test[-1] > 0.9:
        plt.legend(['0.9', 'train', 'test'], loc='upper left')
    else:
        plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, f'FG, Divider {divider}, layers {layers}.png'))
    plt.show()
    print(f'FG, Divider {divider}, layers {layers}, accuracy {accuracy_all_test[-1]}')
    return accuracy_all_train, accuracy_all_test


def evaluate_model(model: Sequential, dividers, learning_type):
    ann_viz(model, title=f"{learning_type}, Layers {len(model.layers), dividers}, dividers inputs")
    # plt.savefig(f"{learning_type}, Layers {len(model.layers)}, {dividers} inputs.png")
    G = convert_to_graph(model)
    # partition = girvan_newman(G)
    # partition = community.best_partition(G)
    # modularity = community.modularity(partition, G)
    # nx.draw(G)
    # plt.show()


def FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, layers, iterations=100, path="./"):
    accuracy_all_test = []
    accuracy_all_train = []
    for col in y_train.columns:
        temp_train, temp_test = FG(X_train, y_train[col], X_test, y_test[col], col, model, layers, iterations=iterations, path=path)
        accuracy_all_train.extend(temp_train)
        accuracy_all_test.extend(temp_test)
    plt.plot(accuracy_all_train)
    plt.plot(accuracy_all_test)
    plt.title(f'FG, Dividers {numbers_to_divide_by}, layers {layers}, full learning')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, 'FG, Dividers {numbers_to_divide_by}, layers {layers}, full learning.png'))
    plt.show()


def main():
    args = crate_parser()
    num_inputs = args.num_inputs
    data_size = args.data_size
    max_number = args.max_number
    learning_type = args.learning_type
    iterations_fg = args.iterations_fg
    iterations_each_round_mvg = args.iterations_each_round_mvg
    amount_of_rounds_mvg = args.amount_of_rounds_mvg
    layers = args.layers

    modules = [3, 7, 11]

    numbers_to_divide_by = get_all_premonition_mult(modules, amount_of_elements=2)

    df: DataFrame = create_data(num_inputs, data_size, max_number, numbers_to_divide_by, modules, path="data")

    print_with_separator("Data Created - Starting to train")
    X = df.filter(regex='^input')
    y = df.filter(regex='^divide_by')
    y_m = df.filter(regex='^m_divide_by')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs("output", exist_ok=True)
    cur_out_dir = os.path.join("output", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(cur_out_dir, exist_ok=True)
    model = model_crate(num_inputs, layers, 4)
    if learning_type == 'MVG':
        MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, layers, num_round=amount_of_rounds_mvg,
            epoch_per_round=iterations_each_round_mvg, path=cur_out_dir)
    elif learning_type == 'FG':
        FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, layers, iterations=iterations_fg, path=cur_out_dir)
    model.save(os.path.join(cur_out_dir, 'model.h5'))

if __name__ == '__main__':
    time = timeit(main, number=1)
    print(f'time to run main: {time}')

#     TODO:
#     layers 3 - 10
#     switch rate 2 - 20
#     modularity
#     start from one number
#
#      Add dipper (2 layers) networks and more layers.
# todo - try 3,5,7 and 3,5,11 and 2,5,7 (multiply of those number).
# todo - learn until near saturation and than swap.

# TODO - to check if the NN memorize the data or not, i.e check if it can.
# TODO - make sure all the data is unique.
# TODO - start from one divider and than add more.
# TODO - check what is minimal layers needed to succeed in validation.
# TODO - check how that history is saved.
# TODO - girvan newman algorithm
# TODO - multicalssfication - multi output - try with loss
# todo - other goals.
# todo - try wider neworks
# todo - add L1
# sequnail lerning of all goals in fg
# find sd and mean of fg
# print number of clsuter in fg
# squre the edges (weights)

# ---- history ----

# for layers in range(5, 6):
#     clear_session()
#     accuracy_all_test = []
#     accuracy_all_train = []
#     model = model_crate(num_inputs, layers)
#     for col in y_train.columns:
#         temp_train, temp_test = FG(X_train, y_train[col], X_test, y_test[col], col, model, layers, iterations=100)
#         accuracy_all_train.extend(temp_train)
#         accuracy_all_test.extend(temp_test)
#     plt.plot(accuracy_all_train)
#     plt.plot(accuracy_all_test)
#     # add regrasion line for the test data to see if it is converging or not
#     m, b = np.polyfit(np.arange(len(accuracy_all_test)), accuracy_all_test, 1)
#     plt.plot(np.arange(len(accuracy_all_test)), m * np.arange(len(accuracy_all_test)) + b)
#     # plt.plot(np.unique(accuracy_all_test), np.poly1d(np.polyfit( np.arange(len(accuracy_all_test)), accuracy_all_test, 1))(np.unique(accuracy_all_test)))
#     plt.title(f'FG, Dividers {numbers_to_divide_by}, layers {layers}, full learning')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(['train', 'test', "regression line"], loc='upper left')
#     plt.savefig(f'FG, Dividers {numbers_to_divide_by}, layers {layers}, full learning.png')
#     plt.show()
#     del model

# for layers in range(0, 7):
#     clear_session()
#     model = model_crate(num_inputs, layers)
#     MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, layers, num_round=8, epoch_per_round=20)
#     # evaluate_model(model, numbers_to_divide_by, 'MVG')
#
#     del model
#
