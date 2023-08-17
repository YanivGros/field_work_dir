# This is a sample Python script.
import argparse
import datetime
import itertools
import logging
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

os.makedirs("output", exist_ok=True)
cur_out_dir = os.path.join("output", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
os.makedirs(cur_out_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, filename=os.path.join(cur_out_dir, "log_file.log"), filemode='w', format='%(message)s')
logger = logging.getLogger(__name__)


def MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model: Sequential, num_round=11, epoch_per_round=10):
    accuracy_columns = ['round', 'epoch', 'divider', 'train_accuracy', 'test_accuracy']
    accuracy_df = pd.DataFrame(columns=accuracy_columns)
    for i in range(num_round):
        for j, number in enumerate(numbers_to_divide_by):
            cur_train = y_train.iloc[:, j]
            cur_test = y_test.iloc[:, j]
            model.fit(X_train, cur_train, validation_data=[X_test, cur_test], epochs=epoch_per_round, verbose=1,
                      shuffle=True,
                      use_multiprocessing=True, workers=32)
            accuracy_df = pd.concat([accuracy_df, pd.DataFrame({'round': i, 'epoch': np.arange(epoch_per_round), 'divider': number,
                                                                'train_accuracy': model.history.history['binary_accuracy'],
                                                                'test_accuracy': model.history.history['val_binary_accuracy']})],
                                    ignore_index=True)
            print(f'round {i} out of {num_round} divider {number}')
        if i % 5 == 0:
            plt.plot(accuracy_df['train_accuracy'])
            plt.plot(accuracy_df['test_accuracy'])
            plt.title(f'MVG epoch {i}, sequential graph')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(cur_out_dir, f'MVG_sequential_step{i}.png'))
            plt.show()
            for divider in numbers_to_divide_by:
                divider_data = accuracy_df[accuracy_df['divider'] == divider]
                plt.plot(divider_data['train_accuracy'].reset_index(drop=True), label=f'train {divider}')
                plt.plot(divider_data['test_accuracy'].reset_index(drop=True), label=f'test {divider}')
            plt.title(f'MVG round {i}, separate graphs')
            plt.legend()
            plt.savefig(
                os.path.join(cur_out_dir,f'MVG_separate_step{i}.png'))
            plt.show()
            accuracy_df.to_csv(os.path.join(cur_out_dir, f'MVG_data.csv'), index=False)
    accuracy_df.to_csv(os.path.join(cur_out_dir, f'MVG_data.csv'), index=False)


def FG(X_train, y_train, X_test, y_test, divider, model: Sequential, iterations=100):
    model.fit(X_train, y_train, epochs=iterations, validation_data=(X_test, y_test), shuffle=True, use_multiprocessing=True, workers=8, )
    accuracy_all_train = model.history.history['binary_accuracy']
    accuracy_all_test = model.history.history['val_binary_accuracy']
    if accuracy_all_test[-1] > 0.9:
        point_where_accuracy_is_09 = bisect.bisect_left(accuracy_all_test, 0.9)
        logger.info(f'point where accuracy is 0.9 is {point_where_accuracy_is_09}')
        plt.plot(point_where_accuracy_is_09, 0.9, 'ro')
        plt.text(point_where_accuracy_is_09, 0.9, f'({point_where_accuracy_is_09}, 0.9)')
    plt.plot(accuracy_all_train)
    plt.plot(accuracy_all_test)
    plt.title(f'FG, Divider {divider}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    if accuracy_all_test[-1] > 0.9:
        plt.legend(['0.9', 'train', 'test'], loc='upper left')
    else:
        plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(cur_out_dir, f'FG_Divider_{divider}.png'))
    plt.show()
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


def FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, iterations=100):
    df_accuracy = pd.DataFrame(columns=['epoch', 'train_accuracy', 'test_accuracy', 'divider'])
    for col in y_train.columns:
        temp_train, temp_test = FG(X_train, y_train[col], X_test, y_test[col], col, model, iterations=iterations)
        df_accuracy = pd.concat([df_accuracy, pd.DataFrame(
            {'epoch': np.arange(len(temp_train)), 'train_accuracy': temp_train, 'test_accuracy': temp_test, 'divider': col})],
                                ignore_index=True)
    plt.plot(df_accuracy['train_accuracy'])
    plt.plot(df_accuracy['test_accuracy'])
    plt.title(f'FG, Dividers {numbers_to_divide_by} full learning')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(cur_out_dir, f'FG_{numbers_to_divide_by}_full_learning.png'))
    plt.show()


def log_with_separator(param):
    logger.info(f'{"-" * 20}{param}{"-" * 20}')


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
    width_multiplayer = args.width_multiplayer
    modules = [3, 5, 11]
    numbers_to_divide_by = get_all_premonition_mult(modules, amount_of_elements=2)
    regularizer = l1(1e-5)

    logger.info(f'num_inputs: {num_inputs}')
    logger.info(f'data_size: {data_size}')
    logger.info(f'max_number: {max_number}')
    logger.info(f'learning_type: {learning_type}')

    if learning_type == 'MVG':
        logger.info(f'iterations_each_round_mvg: {iterations_each_round_mvg}')
        logger.info(f'amount_of_rounds_mvg: {amount_of_rounds_mvg}')
    elif learning_type == 'FG':
        logger.info(f'iterations_fg: {iterations_fg}')
    logger.info(f'modules: {modules}')
    logger.info(f'numbers_to_divide_by: {numbers_to_divide_by}')

    df: DataFrame = create_data(num_inputs, data_size, max_number, numbers_to_divide_by, modules, path="data")
    X = df.filter(regex='^input')
    y = df.filter(regex='^divide_by')
    y_m = df.filter(regex='^m_divide_by')

    log_with_separator("Data information")
    logger.info(f"amount of unique values: {df['decimal'].nunique()}, total amount of values: {len(df['decimal'])}")
    logger.info(f'largest bit in use is {np.ceil(np.log2(df["decimal"].max() + 1))}')
    logger.info(df['decimal'].describe())
    logger.info(y.describe())
    logger.info(y_m.describe())

    plt.hist(df['decimal'], bins=100)
    plt.title('data decimal distribution')
    plt.savefig(os.path.join(cur_out_dir, 'decimal distribution.png'))
    plt.clf()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_crate(num_inputs, layers, width_multiplayer=width_multiplayer, regularizer=regularizer)
    log_with_separator("Model summary")
    logger.info(f'amount of layers: {layers}')
    logger.info(f'layers width: {width_multiplayer * num_inputs}')
    logger.info(f'regularizer: {regularizer.get_config()}')
    model.summary(print_fn=logger.info)
    if learning_type == 'MVG':
        MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, num_round=amount_of_rounds_mvg,
            epoch_per_round=iterations_each_round_mvg)
    elif learning_type == 'FG':
        FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, iterations=iterations_fg)
    model.save(os.path.join(cur_out_dir, 'model.keras'))
    clear_session()


if __name__ == '__main__':
    time = timeit(main, number=1)
    logger.info(f'time to run main: {time}')
    # os.rmdir(cur_out_dir)


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
