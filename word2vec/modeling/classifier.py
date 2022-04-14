from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from utils import settings


def train_classifier(classifier:Any, xtrain=None, ytrain=None):
    clf = classifier(max_depth=2, random_state=0)
    clf.fit(xtrain,ytrain)
    return clf


def train_knn(embeddings,number,save_model_path):
    neigh = NearestNeighbors(n_neighbors=number, p=1,n_jobs=-1)
    neigh.fit(embeddings)
    settings.save_pickle(save_model_path,neigh)


def load_knn(save_model_path):
    knn = settings.load_pickle(save_model_path)
    return knn



