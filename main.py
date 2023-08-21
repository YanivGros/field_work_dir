# This is a sample Python script.
import bisect
import datetime
import logging
import random
from timeit import timeit

import keras
import keras.metrics
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from ann_visualizer.visualize import ann_viz
from keras.backend import clear_session
from keras.models import Sequential
from keras.src.regularizers import l1
from numpy.random import default_rng
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from data.data_crate import create_data
from utility import *

rng = default_rng(seed=42)


def evaluate_model(model, dividers, learning_type):
    ann_viz(model, title=f"{learning_type}, Layers {len(model.layers), dividers}, dividers inputs")
    # plt.savefig(f"{learning_type}, Layers {len(model.layers)}, {dividers} inputs.png")
    G = convert_to_graph(model)
    # partition = girvan_newman(G)
    # partition = community.best_partition(G)
    # modularity = community.modularity(partition, G)
    # nx.draw(G)
    # plt.show()


def MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, num_round=11, epoch_per_round=10, threshold=0.98):
    accuracy_columns = ['round', 'epoch', 'divider', 'train_accuracy', 'test_accuracy']
    accuracy_df = pd.DataFrame(columns=accuracy_columns)
    is_over_threshold = {number: False for number in y_train.columns}
    columns = list(y_train.columns)
    # found = False
    # max_acc = [0 for _ in y_train.columns]
    for i in range(num_round):
        random.shuffle(columns)
        for number in columns:
            cur_train = y_train[number]
            cur_test = y_test[number]
            model.fit(X_train, cur_train, validation_data=[X_test, cur_test], epochs=epoch_per_round, verbose=0,
                      shuffle=True,
                      use_multiprocessing=True, workers=16)
            accuracy_df = pd.concat(
                [accuracy_df, pd.DataFrame({'round': i, 'epoch': np.arange(epoch_per_round), 'divider': int(number.split('_')[-1]),
                                            'train_accuracy': model.history.history['binary_accuracy'],
                                            'test_accuracy': model.history.history['val_binary_accuracy']})],
                ignore_index=True)
            print(f'done round {i} out of {num_round} divider {number} is {model.history.history["val_binary_accuracy"][-1]}')
            if any(num > threshold for num in model.history.history['val_binary_accuracy']) and not is_over_threshold[number]:
                is_over_threshold[number] = True
                print(f'done round {i} out of {num_round} divider {number} is over {threshold}')
            if all(is_over_threshold.values()):
                print(
                    f'learned all dividers in round {i * epoch_per_round + bisect.bisect_left(model.history.history["val_binary_accuracy"], threshold)}')
                accuracy_df.to_csv(os.path.join(cur_out_dir, f'MVG_data_size{len(X_train)}_epoch{epoch_per_round}.csv'), index=False)
                return
                # print(f'learned all dividers in round {i}')
                # if not found:
                #     logger.info(f'learned all dividers in round {i}')
                #     found = True
                # print(f'done! in round {i} out of {num_round} divider {number}')

        # if i % 5 == 0:
        #     plt.plot(accuracy_df['train_accuracy'])
        #     plt.plot(accuracy_df['test_accuracy'])
        #     plt.title(f'MVG epoch {i}, sequential graph')
        #     plt.legend(['train', 'test'])
        #     plt.savefig(os.path.join(cur_out_dir, f'MVG_sequential_step{i}.png'))
        #     plt.show()
        #     for divider in numbers_to_divide_by:
        #         divider_data = accuracy_df[accuracy_df['divider'] == divider]
        #         plt.plot(divider_data['train_accuracy'].reset_index(drop=True), label=f'train {divider}')
        #         plt.plot(divider_data['test_accuracy'].reset_index(drop=True), label=f'test {divider}')
        #         plt.title(f'MVG round {i}, separate graphs divider {divider}')
        #         plt.legend()
        #         plt.savefig(
        #             os.path.join(cur_out_dir, f'MVG_separate_step{i}_divider{divider}.png'))
        #         plt.show()
        #     accuracy_df.to_csv(os.path.join(cur_out_dir, f'MVG_data.csv'), index=False)
    accuracy_df.to_csv(os.path.join(cur_out_dir, f'MVG_data_size{len(X_train)}_epoch{epoch_per_round}.csv'), index=False)


class StopAtThresholdCallBack(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(StopAtThresholdCallBack, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["val_binary_accuracy"]
        if accuracy >= self.threshold:
            self.model.stop_training = True


def FG(X_train, y_train, X_test, y_test, divider, model, iterations=100, threshold=0.99):
    # make the model stoped when it reach 0.99 accuracy
    # call_back = StopAtThresholdCallBack(threshold)
    call_back = keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=200,start_from_epoch=300, verbose=0,
                                              mode='auto', baseline=None, restore_best_weights=False)
    model.fit(X_train, y_train, epochs=iterations, validation_data=(X_test, y_test), shuffle=True, use_multiprocessing=True, workers=16,
              callbacks=[call_back], verbose=1)
    print(
        f"divider {divider} stopped at {len(model.history.history['val_binary_accuracy']), model.history.history['val_binary_accuracy'][-1]}")
    accuracy_all_train = model.history.history['binary_accuracy']
    accuracy_all_test = model.history.history['val_binary_accuracy']
    found = False
    if any(num > threshold for num in accuracy_all_test):
        threshold_point = accuracy_all_test.index(next((num for num in accuracy_all_test if num > threshold), None))
        logger.info(f'point where accuracy is {threshold} is {threshold_point}')
        plt.plot(threshold_point, threshold, 'ro')
        plt.text(threshold_point, threshold, f'({threshold_point},{threshold}')
        found = True
    plt.plot(accuracy_all_train)
    plt.plot(accuracy_all_test)
    plt.title(f'FG, Divider {divider} train size {len(X_train)}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    if found:
        plt.legend([f'f{threshold}', 'train', 'test'])
    else:
        plt.legend(['train', 'test'])
    plt.savefig(os.path.join(cur_out_dir, f'FG_Divider_{divider}.png'))
    plt.show()
    return accuracy_all_train, accuracy_all_test


def FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, iterations=100):
    df_accuracy = pd.DataFrame(columns=['epoch', 'train_accuracy', 'test_accuracy', 'divider'])
    for col in y_train.columns:
        temp_train, temp_test = FG(X_train, y_train[col], X_test, y_test[col], col, model, iterations=iterations)
        df_accuracy = pd.concat([df_accuracy, pd.DataFrame(
            {'epoch': np.arange(len(temp_train)), 'train_accuracy': temp_train, 'test_accuracy': temp_test, 'divider': col})],
                                ignore_index=True)
        print(
            f"divider {col} max acc {max(temp_test)} at epoch {temp_test.index(max(temp_test))}")
    plt.plot(df_accuracy['train_accuracy'])
    plt.plot(df_accuracy['test_accuracy'])
    plt.title(f'FG, Dividers {numbers_to_divide_by} train size {len(X_train)} full learning')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.savefig(os.path.join(cur_out_dir, f'FG_{numbers_to_divide_by}_training_size_{len(X_train)}_full_learning.png'))
    df_accuracy.to_csv(os.path.join(cur_out_dir, f'FG_{numbers_to_divide_by}_full_learning.csv'), index=False)
    plt.show()


def log_with_separator(param):
    logger.info(f'{"-" * 20}{param}{"-" * 20}')


logger: logging.Logger = logging.getLogger(__name__)
cur_out_dir: str = "./"


def main():
    os.makedirs("output", exist_ok=True)
    global cur_out_dir
    cur_out_dir = os.path.join("output", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(cur_out_dir, exist_ok=True)

    global logger
    logging.basicConfig(level=logging.INFO, filename=os.path.join(cur_out_dir, "log_file.log"), filemode='w', format='%(message)s')
    logger = logging.getLogger(__name__)
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
    modules = [3, 7, 11]
    numbers_to_divide_by = get_all_premonition_mult(modules, amount_of_elements=2)
    regularizer = l1(1e-8)
    # regularizer = None

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
    for data_size in range(5000, 20_001, 5000):
        df: DataFrame = create_data(num_inputs, data_size, max_number, numbers_to_divide_by, modules, path="data")
        X = df.filter(regex='^input')
        y = df.filter(regex='^divide_by')
        y_m = df.filter(regex='^m_divide_by')

        log_with_separator("Data information")
        logger.info(f"amount of unique values: {df['decimal'].nunique()}, total amount of values: {len(df['decimal'])}")
        # logger.info(f'largest bit in use is {np.ceil(np.log2(df["decimal"].max() + 1))}')
        # logger.info(df['decimal'].describe())
        logger.info(y.describe())
        # logger.info(y_m.describe())
        # plt.hist(df['decimal'], bins=100)
        # plt.title(f'data decimal distribution, data size {data_size}')
        # plt.savefig(os.path.join(cur_out_dir, f'decimal_distribution_size_{data_size}.png'))
        # plt.show()
        # plt.clf()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for epoch_per_round in [20]:
            print(f'data size {data_size}, epoch per round {epoch_per_round}')
            model = model_crate(num_inputs, layers, width_multiplayer=width_multiplayer, regularizer=regularizer)
            # if regularizer is not None:
            # logger.info(f'regularizer: {regularizer.get_config()}')
            # else:
            #     logger.info(f'regularizer: None')
            # log_with_separator("Model summary")
            # logger.info(f'amount of layers: {layers}')
            # logger.info(f'layers width: {width_multiplayer * num_inputs}')
            # model.summary(print_fn=logger.info)
            if learning_type == 'MVG':
                MVG(X_train, y_train, X_test, y_test, numbers_to_divide_by, model ,
                    epoch_per_round=epoch_per_round)
            elif learning_type == 'FG':
                FG_all(X_train, y_train, X_test, y_test, numbers_to_divide_by, model, iterations=iterations_fg)
            model.save(os.path.join(cur_out_dir, f'MVG_data_size{len(X_train)}_epoch{epoch_per_round}.keras'))
            clear_session()
            del model


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
# TODO - multicalssfication - multi output_old - try with loss
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
