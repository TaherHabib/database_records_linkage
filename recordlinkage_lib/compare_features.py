import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import recordlinkage
from recordlinkage.preprocessing import clean
from utils.settings import get_project_root

ROOT = get_project_root()


def  compute_feature_compare(df_left=None,
                             df_right=None,
                             index_algorithm='block',
                             left_block_cols=None,
                             right_block_cols=None,
                             left_sorting_key=None,
                             right_sorting_key=None,
                             similarity_metric='cosine',
                             similarity_threshold=0.8,
                             missing_value=-1):

    indexer = recordlinkage.Index()
    if index_algorithm == 'block':
        indexer.block(left_on=left_block_cols, right_on=right_block_cols)
    elif index_algorithm == 'sorted_neighborhood':
        indexer.sortedneighbourhood(left_on=left_sorting_key, right_on=right_sorting_key,
                                    block_left_on=left_block_cols, block_right_on=right_block_cols)
    else:
        raise ValueError('Please choose a valid indexing algorithm from: \'block\' or \'sorted_neighborhood\'.')

    candidates = indexer.index(x=df_left, x_link=df_right)
    print('Number of candidates for comparison: {}'.format(len(candidates)))

    comp = recordlinkage.Compare(n_jobs=1)
    comp.string(left_on='name', right_on='name', label='name', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='street_name', right_on='street_name', label='street_name', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='street_type', right_on='street_type', label='street_type', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='street_name', right_on='street_name', label='street_name', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='city', right_on='city', label='city', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.exact(left_on='street_number', right_on='street_number', label='street_number', missing_value=missing_value)
    comp.exact(left_on='postal_code', right_on='postal_code', label='postal_code', missing_value=missing_value)

    df_feature_compare = comp.compute(pairs=candidates, x=df_left, x_link=df_right)

    return df_feature_compare


if __name__ == '__main__':

    left_block_on_cols = right_block_on_cols = ['city', 'name']
    index_algorithm = 'block' # 'sorted_neighborhood'
    # left_sorting_key = right_sorting_key = ['name']
    similarity_metric = 'cosine' # 'jarowinkler'
    threshold = 0.8
    label_for_missing_values = -1

    s1_cstr = pd.read_csv(os.path.join(ROOT, 'data', 'source1_cstr.csv'))
    s2_cstr_parsed = pd.read_csv(os.path.join(ROOT, 'data', 'source2_cstr_parsedaddress.csv'))

    df_feature_compare = compute_feature_compare(df_left=s1_cstr,
                                                 df_right=s2_cstr_parsed,
                                                 index_algorithm=index_algorithm,
                                                 left_block_cols=left_block_on_cols,
                                                 right_block_cols=right_block_on_cols,
                                                 #left_sorting_key=left_sorting_key,
                                                 #right_sorting_key=right_sorting_key,
                                                 similarity_metric=similarity_metric,
                                                 similarity_threshold=threshold,
                                                 missing_value=label_for_missing_values)

    df_feature_compare.to_csv(os.path.join(ROOT, 'data', 'compared_features_{}_{}_{}'.format(index_algorithm,
                                                                                             similarity_metric,
                                                                                             threshold
                                                                                             )
                                           )
                              )
