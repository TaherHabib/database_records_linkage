from utils import settings
import os
import pandas as pd
from word2vec.modeling import word_2_vec
from word2vec.modeling.classifier import load_knn
import swifter
from preprocessing import pandas_oper
import glob

project_root = settings.get_project_root()


def split_dataframes(dataframe_2):
    pandas_oper.sampling_dataframe(dataframe_2,36,os.path.join(project_root, 'data', 'dataset'))


def merge_dataframes(df,word2vec_model,knn_model,embedding):
    address = df['address']
    vector = word_2_vec.inference(word2vec_model,address)
    distances, indices = knn_model.kneighbors([vector])
    id = int(embedding[indices[0][0]][0])
    return id


def main():
    '''

    The main() creates predictions of potential matches using embeddings of source2's 'address' columns and finding
    its nearest neighbor source1 'address' embedding. The ID from source1's match is stored for downstream evaluation.

    :return:
    '''

    embeddings_path = os.path.join(project_root, 'data', 'dataset', 's1_embeddings.npz')
    w2v_output = os.path.join(project_root, 'data', 'trained_data', 'source1_model.model')
    knn_path = os.path.join(project_root, 'data', 'trained_data', 'knn_1.pickle')

    word2vec_model = word_2_vec.load_w2v(w2v_output)
    embeddings = word_2_vec.load_npz(embeddings_path)
    knn_model = load_knn(knn_path)

    dataframe_2 = pd.read_csv(os.path.join(project_root, 'data', 'dataset', 'address_2_1.csv'))
    split_dataframes(dataframe_2)
    csv_files = glob.glob(os.path.join(os.path.join(project_root, 'data', 'dataset', 'source_address2_split_*')))

    for csv in csv_files:
        df = pd.read_csv(csv)
        df["id"] = df.swifter.apply(merge_dataframes, args=(word2vec_model,knn_model,embeddings,), axis=1)
        final_output_dataframe = os.path.join(project_root, 'data', 'dataset', 'final_output',csv.split('/')[-1])
        df.to_csv(final_output_dataframe,index=False)
        os.remove(csv)

    csv_files = glob.glob(os.path.join(os.path.join(project_root, 'data', 'dataset', 'final_output', 'source_address2_split_*')))
    df = pd.read_csv(os.path.join(os.path.join(project_root, 'data', 'dataset', 'final_output', 'source_address2_split_0.csv')))
    for csv in csv_files:
        if csv.split('/')[-1].split('_')[-1][0] != '0':
            df = df.append(pd.read_csv(csv), ignore_index=True)

    # saving one final output
    df.to_csv(os.path.join(project_root, 'data', 'dataset', 'final_output', 'all_source2_address_ids.csv'), index=False)

