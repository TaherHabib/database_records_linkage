from pathlib import Path
import os
import pickle


def get_project_root():
    '''
    for setting the root of the project independent of OS.
    :return: root of the project
    '''

    return os.path.abspath(Path(__file__).parent.parent)


def save_pickle(outpath, data):
    """(Over)write data to new pickle file."""
    #outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {outpath}')


def load_pickle(inpath):
    print(f'Loading from existing pickle file... {inpath}')
    with open(inpath, "rb") as f:
        return pickle.load(f)
