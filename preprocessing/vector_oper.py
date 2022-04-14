import numpy as np
import random

def shuffle_column(dataframe,column):
    """

    :return:
    """
    fraction = 0.6
    n_rows = len(dataframe)
    n_shuffle = int(n_rows * fraction)
    indexed_rows = random.sample(range(1, n_rows), n_shuffle)

    dataframe.loc[indexed_rows, column] = np.random.permutation(dataframe.loc[indexed_rows, column])

    return dataframe,indexed_rows
