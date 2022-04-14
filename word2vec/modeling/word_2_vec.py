import pandas as pd
import logging
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
# nltk.download('book')
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm
from preprocessing import vector_oper

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def train_w2v(dataframe,model_out):
    # get array of addresses and train Word2vec
    source1_address = dataframe['source1_address'].values.tolist()

    # tokenize the each address
    tok_source1_address = [word_tokenize(address) for i,address in enumerate(tqdm(source1_address))]

    # refer to here for all parameters:
    # https://radimrehurek.com/gensim/models/word2vec.html
    source1_model = Word2Vec(tok_source1_address, sg=1, vector_size=200, window=5, min_count=1, workers=4, epochs=400)

    # save model to file
    source1_model.save(model_out)


def load_w2v(model_path):
    """
     return w2v
    :param model:
    :return:
    """
    model = Word2Vec.load(model_path)

    return model


def get_vector(model,sentence):
    vectors = [model.wv[w] for w in word_tokenize(sentence) if w in model.wv]

    v = np.zeros(model.vector_size)

    if len(vectors) > 0:
        v = (np.array([sum(x) for x in zip(*vectors)])) / v.size

    return v


def similarity(add1, add2):
    xv = add1
    yv = add2
    score = 0

    if xv.size > 0 and yv.size > 0:
        score = dot(xv, yv) / (norm(xv) * norm(yv))

    return score


def inference(model,sentence):
    return get_vector(model, sentence)


def create_positive_examples(model,dataframe,label=True):

    flag = -1
    positive_samples = pd.DataFrame(columns=['id'] + ['feature_{}'.format(x) for x in range(1, 401)] + ['label'])
    if label:
        print("Creating positive samples")
        flag = 1
    else:
        print("Creating Negative samples")
        flag = 0
    i = 0
    for index, row in tqdm(dataframe.iterrows()):
        source1_vector = inference(model,row['source1_address'])
        source2_vector = inference(model,row['address'])
        new_row = np.concatenate(([row['id']],source1_vector, source2_vector,[flag]))
        positive_samples.loc[index] = new_row

        i = i+1
        if i >= 100000:
            break

    return positive_samples


def create_negative_examples(model,dataframe):
    dataframe,indexed_rows = vector_oper.shuffle_column(dataframe,'address')

    # only copy those indexes which are shuffled
    dataframe = pd.DataFrame(dataframe,index=indexed_rows)

    # reset index
    dataframe.reset_index(drop=True)

    negative_samples = create_positive_examples(model,dataframe,False)

    return negative_samples


def create_numpy_embeddings(model,dataframe,source='source1_address'):
    source1_embeddings = []
    for index, row in dataframe.iterrows():
        vector = inference(model, row[source])
        embedding_with_id = np.concatenate(([row['id']], vector))
        source1_embeddings.append(embedding_with_id.tolist())

    return np.array(source1_embeddings)


def get_vectorise_dataset(model,dataframe, format='numpy',column_name_npy='source1_address'):
    if format == 'csv':
        positive = create_positive_examples(model,dataframe)
        negative = create_negative_examples(model,dataframe)
        dataset = positive.append(negative, ignore_index=True)

    elif format == 'numpy':
        dataset = create_numpy_embeddings(model,dataframe, source=column_name_npy)

    else:
        dataset = None

    return dataset


def load_npz(model_path):
    embeddings = np.load(model_path)
    return embeddings['arr_0']











