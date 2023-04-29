# This is a sample Python script.
import keras.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers.optimizer_v2.adam import Adam

from sklearn.model_selection import train_test_split

from numpy.random import default_rng

from data.data_crate import crate_data

rng = default_rng(seed=42)
# todo - add dipper (2 layers) networks and more layers.
# todo - try 3,5,7 and 3,5,11 and 2,5,7 (multiply of those number).
# todo - learn until near saturation and than swap.



if __name__ == '__main__':
    # load data from csv

    num_inputs = 32
    sample_size = 10000
    # df = crate_data(num_inputs, sample_size, [3 * (7 ** 2), (3 ** 2) * 7, (3 ** 2) * (7 ** 2)])
    df = crate_data(num_inputs, sample_size, [3,5,7])
    df.drop(['decimal'], axis=1, inplace=True)

    # create the input and output data
    X = df.iloc[:, 0:num_inputs].values
    y = df.iloc[:, num_inputs:].values
    divide_name = df.columns[num_inputs:]

    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(num_inputs, input_dim=num_inputs, activation='relu'))
    model.add(Dense(num_inputs, activation='relu'))
    model.add(Dense(num_inputs, activation='relu'))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[keras.metrics.BinaryAccuracy()])

    # divide to train and test
    accuracy = {}
    for divider in divide_name:
        accuracy[divider] = []
    accuracy_all = []
    for i in range(100):
        cur_y_train = y_train[:, i % len(divide_name)]
        cur_y_test = y_test[:, i % len(divide_name)]
        # Train the model on the current desired output
        # model.train_on_batch(X_train, cur_y_train)
        accuracy[divide_name[i % len(divide_name)]].append(
            model.fit(X_train, cur_y_train, epochs=1).history['binary_accuracy'][0])
        # accuracy_all.append(model.fit(X_train, cur_y_train, epochs=1).history['binary_accuracy'][0])
        # print("Finished training on " + divide_name[i % len(divide_name)])
        # accuracy[divide_name[i % len(divide_name)]].append(model.evaluate(X_train, cur_y_train, verbose=0)[1])
        print(i)

    # plot the results
    for divider in divide_name:
        plt.plot(accuracy[divider], label=divider)
    # plt.plot(accuracy_all, label='all')
    plt.legend()
    plt.show()
    exit(0)

    for i in range(len(divide_name)):
        model = Sequential()
        model.add(Dense(num_inputs, input_dim=num_inputs, activation='relu'))
        model.add(Dense(num_inputs, activation='relu'))
        model.add(Dense(num_inputs, activation='relu'))
        model.add(Dense(X_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[keras.metrics.BinaryAccuracy()])
        # record accuracy after each epoch
        acc_list = []
        a = model.fit(X_train, y_train[:, i], epochs=100, workers=4, use_multiprocessing=True)
        plt.plot(a.history['binary_accuracy'], label=divide_name[i])
        plt.legend()
        plt.show()
        print(model.summary())

        print(model.evaluate(X_test, y_test[:, i], verbose=0)[1])
