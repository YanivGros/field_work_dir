import numpy as np
import pandas as pd


def create_data(num_bits, num_samples, max_number, numbers_to_divide_by: list, save=False, use_old_data=False) -> pd.DataFrame:
    """
    :param num_bits: The number of bits to represent the input
    :param num_samples: The number of samples to create
    :param max_number: The maximum number to represent in the input
    :param numbers_to_divide_by: The numbers to divide by (the output)
    :param save: Whether to save the data to a file
    :param use_old_data: Whether to use old data from a file
    :return: A dataframe with the input and output
    """
    if use_old_data:
        try:
            df = pd.read_csv(f"data_max_{max_number}_sample_{num_samples}_first_divider_{numbers_to_divide_by}.csv")
            print(
                f"loaded data from file data_max_{max_number}_sample_{num_samples}_first_divider_{numbers_to_divide_by}.csv")
            return df
        except FileNotFoundError:
            print(
                f"could not find file data_max_{max_number}_sample_{num_samples}_first_divider_{numbers_to_divide_by}.csv")
            print("exiting...")
            exit(1)
    assert max(numbers_to_divide_by) * max_number < 2 ** (num_bits - 1), "not enough bits to represent all numbers"
    decimal_numbers = np.random.randint(low=0, high=max_number, size=num_samples) * np.random.choice(
        numbers_to_divide_by, num_samples)
    binary_numbers = np.unpackbits(decimal_numbers.view(np.uint8)).reshape(-1, num_bits)
    binary_numbers = binary_numbers.astype(int)
    df = pd.DataFrame(binary_numbers)
    df.columns = [f'input_{i}' for i in range(num_bits)]
    df['decimal'] = decimal_numbers
    for divider in numbers_to_divide_by:
        df[f"divide_by_{str(divider)}"] = (decimal_numbers % divider == 0).astype(int)
    if save:
        df.to_csv(f"data_max_{max_number}_sample_{num_samples}_first_divider_{numbers_to_divide_by}.csv", index=False)
    return df


if __name__ == '__main__':
    create_data(32, 10000, 100000, [4, 6, 9], save=True)
