import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def clean_data(df: pd.DataFrame):
    mean = df['decimal'].mean()
    var = df['decimal']
    df.drop((df[df['decimal'] > mean + 2 * np.sqrt(var)]).index, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    largest_bit = np.ceil(np.log2(df['decimal'].max() + 1))
    print(f"amount of unique values: {df['decimal'].nunique()}, total amount of values: {len(df['decimal'])}")
    print(f'largest bit in use is {largest_bit}')
    print(df['decimal'].describe())
    print(df.filter(regex='^divide_by').describe())
    plt.hist(df['decimal'], bins=100)
    plt.title('decimal distribution')
    plt.savefig('decimal distribution.png')
    plt.show()
    print(df.filter(regex='^m_divide_by').describe())
    return df


def create_data(num_bits, num_samples, max_number, numbers_to_divide_by: list, modules: list, path: str = "./") -> pd.DataFrame:
    """
    :param path: The path to save the data
    :param num_bits: The number of bits to represent the input
    :param num_samples: The number of samples to create
    :param max_number: The maximum number to represent in the input
    :param numbers_to_divide_by: The numbers to divide by (the output)
    :param modules: The modules that build the numbers to divide by
    :return: A dataframe with the input and output
    """
    file_path = os.path.join(path, f"data_max_{max_number}_sample_{num_samples}_dividers_{numbers_to_divide_by}.csv")
    try:
        df = pd.read_csv(file_path)
        print(f"loaded data from file {file_path}")
        return df
    except FileNotFoundError:
        print(f"could not find file {file_path}")
        print("creating new data")
    assert max(numbers_to_divide_by) * max_number < 2 ** (num_bits - 1), "not enough bits to represent all numbers"
    decimal_numbers = np.random.randint(low=0, high=max_number, size=num_samples) * np.random.choice(
        numbers_to_divide_by, num_samples)
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
    return clean_data(df)


if __name__ == '__main__':
    create_data(32, 10000, 100000, [4, 6, 9], modules=[2, 3])
