from typing import List

import pandas as pd
from pandas import DataFrame as df
from recordlinkage.preprocessing import clean
import numpy as np
import os


def dtype_float_int_to_str(dataframe:df) -> df:

    # change all float to int and then to string
    columns = list(dataframe.select_dtypes(include=['float']).columns)
    for column in columns:
        dataframe[column] = dataframe[column].fillna(0).astype(int).astype(str)
        dataframe[column] = dataframe[column].replace(['0'], ' ')

    return dataframe


def filter_columns(dataframe:df,columns:np.array)->df:

    dataframe = dataframe.drop(columns=columns)
    return dataframe


def merge_columns(dataframe:df,columns:np.array,new_column_name:str)->df:

    dataframe[new_column_name] = dataframe[columns].apply(lambda row: ' '.join(row.fillna('').values.astype(str)), axis=1)
    return dataframe


def drop_null_row(dataframe:df)->df:

    dataframe.dropna(inplace=True)

    return dataframe


def clean_columns(dataframe:df, columns:np.array)->df:

    for column in columns:
        dataframe[column] = clean(dataframe[column], strip_accents='unicode')

    return dataframe


def sampling_dataframe(dataframe, n, output_path):

    length = len(dataframe)
    start = 0
    range_ = int(length/n)
    stop = range_
    csv_name = 'source_address2_split_'
    for i in range(n):
        if i < n-1:
            df_ = dataframe.loc[start:stop]
        else:
            df_ = dataframe.loc[start:length-1]
        start = stop+1
        stop = stop + range_
        df_.to_csv(os.path.join(output_path,csv_name+str(i)+'.csv'), index=False)

    return True


