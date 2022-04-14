import os
from preprocessing import pandas_oper
import pandas as pd
import numpy as np
from utils import settings


def prepare_dataframe(file):
    columns_to_merge = np.array(['address_line2',
                                 'street_number',
                                 'street_type',
                                 'street_name',
                                 'postal_code',
                                 'city'])

    columns_to_drop = np.concatenate((columns_to_merge,np.array(['name_y', 'website', 'name_x', 'Unnamed: 0'])))

    df = (pd.read_csv(file,sep='\t')
          .pipe(pandas_oper.dtype_float_int_to_str)
          .pipe(pandas_oper.merge_columns,columns=columns_to_merge,new_column_name='source1_address')
          .pipe(pandas_oper.filter_columns,columns=columns_to_drop)
          .pipe(pandas_oper.drop_null_row) # 700 rows approx
          .pipe(pandas_oper.clean_columns,columns=['source1_address','address']))

    return df


def main():
    project_root = settings.get_project_root()
    file = os.path.join(project_root, 'data', 's1_s2_commonids.tsv')
    df = prepare_dataframe(file)
    df.to_csv(os.path.join(project_root, 'data', 'dataset', 's1_s2_commonids_preprocessed.csv'))

