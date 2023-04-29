import numpy as np
import pandas as pd


def crate_data(num_bits, num_samples, numbers_to_divide_by: list):

    max_number = 10000
    decimal_numbers = np.random.randint(0, max_number, num_samples) * np.random.choice(numbers_to_divide_by, num_samples)
    binary_numbers = np.unpackbits(decimal_numbers.view(np.uint8)).reshape(-1, num_bits)
    binary_numbers = binary_numbers.astype(int)
    df = pd.DataFrame(binary_numbers)
    df['decimal'] = decimal_numbers
    for divider in numbers_to_divide_by:
        df[f"divide_by_{str(divider)}"] = (decimal_numbers % divider == 0).astype(int)
    # df.to_csv(f"data_{num_bits}_{num_samples}_first_divider_{numbers_to_divide_by[0]}.csv", index=False)
    return df
    # num_inputs = num_bits
    # inputs_binary = np.random.randint(2, size=(num_samples, num_inputs))
    # inputs_decimal = np.array([int(''.join(str(x) for x in input), 2) for input in inputs_binary])
    # df = pd.DataFrame(np.concatenate((inputs_binary, inputs_decimal.reshape(-1, 1)), axis=1),
    #                   columns=[f'input_{i}' for i in range(num_inputs)] + ['decimal'])
    #
    # for num in numbers_to_divide_by:
    #     df[f'divide_by_{num}'] = (inputs_decimal % num == 0).astype(int)
    # # save to csv
    # df.to_csv(f'data_{num_bits}_{num_samples}.csv', index=False)

if __name__ == '__main__':
    crate_data(32, 10000, [4, 6, 9])