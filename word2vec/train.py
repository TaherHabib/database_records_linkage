import os
from utils import settings
from word2vec.modeling import word_2_vec
import pandas as pd
import numpy as np
from word2vec.modeling import classifier
project_root = settings.get_project_root()
w2v_output = os.path.join(project_root, 'data', 'trained_data', 'source1_model.model')
knn_path = os.path.join(project_root, 'data', 'trained_data', 'knn_1.pickle')
embeddings_path = os.path.join(project_root, 'data', 'dataset', 's1_embeddings.npz')


def trainw2v():
    df = pd.read_csv(os.path.join(project_root, 'data', 'dataset', 's1_s2_commonids_preprocessed.csv'))
    word_2_vec.train_w2v(df, w2v_output)


def generate_dataset():
    df = pd.read_csv(os.path.join(project_root, 'data',  'dataset', 's1_s2_commonids_preprocessed.csv'))
    word2vec_model = word_2_vec.load_w2v(w2v_output)
    format_ = 'numpy'
    # get vectorised pandas dataset
    vec_df = word_2_vec.get_vectorise_dataset(word2vec_model, df, format=format_,
                                              column_name_npy='source1_address')
    if format_ == 'csv':
        vec_df.to_csv(os.path.join(project_root, 'data', 'dataset', 's1_s2_concat_embeddings_labels.csv'))
    else:
        np.savez_compressed(embeddings_path, vec_df)


def train_knn():
    embeddings = word_2_vec.load_npz(embeddings_path)
    classifier.train_knn(embeddings[:, 1:], number=2, save_model_path=knn_path)


def main():
    trainw2v()
    generate_dataset()
    train_knn()
