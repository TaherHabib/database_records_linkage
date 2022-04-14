import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import recordlinkage
from recordlinkage.preprocessing import clean
from utils.settings import get_project_root

ROOT = get_project_root()


def compute_feature_compare(df_left=None,
                            df_right=None,
                            index_algorithm=None,
                            index_algorithm_params=None,
                            similarity_metric=None,
                            similarity_threshold=None,
                            missing_value=-1):

    indexer = recordlinkage.Index()
    if index_algorithm == 'block':
        indexer.block(left_on=index_algorithm_params['left_block_on_cols'],
                      right_on=index_algorithm_params['right_block_on_cols'])
    elif index_algorithm == 'sorted_neighborhood':
        indexer.sortedneighbourhood(left_on=index_algorithm_params['left_sorting_key'],
                                    right_on=index_algorithm_params['right_sorting_key'],
                                    block_left_on=index_algorithm_params['left_block_on_cols'],
                                    block_right_on=index_algorithm_params['right_block_on_cols'])
    else:
        raise ValueError('Please choose a valid indexing algorithm from: \'block\' or \'sorted_neighborhood\'.')

    candidates = indexer.index(x=df_left, x_link=df_right)
    print('Number of candidates for comparison: {}'.format(len(candidates)))

    comp = recordlinkage.Compare(n_jobs=2)
    # comp.VariableA(on='id', normalize=False, missing_value=np.nan, label='id_source1')
    # comp.VariableB(on='id', normalize=False, missing_value=np.nan, label='id_source2')
    comp.string(left_on='name', right_on='name', label='name', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='street_name', right_on='street_name', label='street_name', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='street_type', right_on='street_type', label='street_type', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='address_line2', right_on='address_line2', label='address_line2', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.string(left_on='city', right_on='city', label='city', missing_value=missing_value,
                threshold=similarity_threshold, method=similarity_metric)
    comp.exact(left_on='street_number', right_on='street_number', label='street_number', missing_value=missing_value)
    comp.exact(left_on='postal_code', right_on='postal_code', label='postal_code', missing_value=missing_value)

    df_feature_compare = comp.compute(pairs=candidates, x=df_left, x_link=df_right)

    return df_feature_compare


if __name__ == '__main__':

    index_algorithm = 'block' # 'sorted_neighborhood' #

    if index_algorithm == 'sorted_neighborhood':
        index_algorithm_params = {
            'left_block_on_cols': ['id'], #['city'],
            'right_block_on_cols': ['id'], #['city'],
            'left_sorting_key': 'name',
            'right_sorting_key': 'name',
        }
    else:
        index_algorithm_params = {
            'left_block_on_cols': ['id'], #['city', 'name'],
            'right_block_on_cols': ['id'], #['city', 'name'],
        }

    similarity_metric = 'jarowinkler' # 'cosine'
    threshold = 0.85

    label_for_missing_values = 0

    s1_cstr = pd.read_csv(os.path.join(ROOT, 'data', 'source1_cstr.csv'))
    s2_cstr_parsed = pd.read_csv(os.path.join(ROOT, 'data', 'source2_cstr_parsedaddress.csv'))

    df_feature_compare = compute_feature_compare(df_left=s1_cstr,
                                                 df_right=s2_cstr_parsed,
                                                 index_algorithm=index_algorithm,
                                                 index_algorithm_params=index_algorithm_params,
                                                 similarity_metric=similarity_metric,
                                                 similarity_threshold=threshold,
                                                 missing_value=label_for_missing_values)

    df_feature_compare.rename(columns={'Unnamed: 0': 'Source1_index', 'Unnamed: 1': 'Source2_index'}, inplace=True)
    df_feature_compare.to_csv(os.path.join(ROOT, 'data', 'compared_features_id_{}_{}_{}.csv'.format(index_algorithm,
                                                                                                    similarity_metric,
                                                                                                    threshold
                                                                                                    )
                                           )
                              )

