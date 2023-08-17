import pandas as pd
from matplotlib import pyplot as plt


def main():
    accuracy_df = pd.read_csv("data.csv")
    layers = 5
    epoch_per_round = 40
    numbers_to_divide_by = [21,33,77]
    plt.figure(figsize=(30, 15))
    plt.plot(accuracy_df['train_accuracy'])
    plt.plot(accuracy_df['test_accuracy'])
    plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, divider {numbers_to_divide_by}')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("first.png")
    plt.show()
    plt.figure(figsize=(30, 15))
    for divider in numbers_to_divide_by:
        divider_data = accuracy_df[accuracy_df['divider'] == divider]
        plt.plot(divider_data['train_accuracy'].reset_index(drop=True), label=f'train {divider}')
        plt.plot(divider_data['test_accuracy'].reset_index(drop=True), label=f'test {divider}')

    plt.title(f'MVG, layers {layers}, round epoch {epoch_per_round}, {numbers_to_divide_by}, separate graphs')
    plt.legend()
    plt.savefig("second.png")
    plt.show()

if __name__ == '__main__':
    main()