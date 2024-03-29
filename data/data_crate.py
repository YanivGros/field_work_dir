import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def clean_data(df: pd.DataFrame):
    """
    Remove outliers and duplicates from the data
    :param df: dataframe to clean
    :return: the cleaned dataframe
    """
    # mean = df['decimal'].mean()
    # std = df['decimal'].std()
    # df.drop((df[df['decimal'] > mean + std]).index, inplace=True)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.describe())
    return df


def create_data(num_bits, num_samples, max_number, numbers_to_divide_by, modules, path="./") -> pd.DataFrame:
    """
    :param path: The path to save the data
    :param num_bits: The number of bits to represent the input
    :param num_samples: The number of samples to create
    :param max_number: The maximum number to represent in the input
    :param numbers_to_divide_by: The numbers to divide by (the output_old)
    :param modules: The modules that build the numbers to divide by
    :return: A dataframe with the input and output_old
    """
    file_path = os.path.join(path, f"data_max_{max_number}_sample_{num_samples}_dividers_{numbers_to_divide_by}.csv")
    try:
        df = pd.read_csv(file_path)
        print(f"loaded data from file {file_path}")
        return df
    except FileNotFoundError:
        print(f"could not find file {file_path}")
        print("creating new data")
    assert max_number < 2 ** (num_bits - 1), "not enough bits to represent all numbers"
    # round the number to the nearest number that divide by a number in numbers_to_divide_by
    decimal_numbers = np.random.randint(low=0, high=max_number, size=num_samples)
    random_choice = np.random.choice(numbers_to_divide_by, num_samples)
    decimal_numbers = np.round(decimal_numbers / random_choice).astype(int) * random_choice
    # decimal_numbers = np.random.randint(low=0, high=max_number, size=num_samples) * np.random.choice(
    #     numbers_to_divide_by, num_samples)
    binary_repr_func = np.vectorize(lambda num: list(map(int, np.binary_repr(num).zfill(num_bits))), otypes=[np.ndarray])
    binary_numbers = np.array(binary_repr_func(decimal_numbers).tolist())
    df = pd.DataFrame(binary_numbers)
    df.columns = [f'input_{i}' for i in range(num_bits, 0, -1)]
    df['decimal'] = decimal_numbers
    for divider in numbers_to_divide_by:
        df[f"divide_by_{str(divider)}"] = (decimal_numbers % divider == 0).astype(int)
    for divider in modules:
        df[f"m_divide_by_{str(divider)}"] = (decimal_numbers % divider == 0).astype(int)
    cleaned_df = clean_data(df)
    cleaned_df.to_csv(file_path, index=False)
    return cleaned_df


if __name__ == '__main__':
    create_data(32, 10000, 100000, [4, 6, 9], modules=[2, 3])
