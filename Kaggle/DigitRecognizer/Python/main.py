#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas     # for read_csv function
import numpy
import time

from sklearn.decomposition import PCA

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

TRAIN_FILE = "../data/train.csv"
TEST_FILE  = "../data/test.csv"
NB_SAMPLES = 2000 # 42000


##########################
# Input & output data,
##########################
def readTrainingSet():
    """
        Read CSV file
        :return:
        - m : number of data samples
        - n : number of features (784)
        - X : array of M x N : input data
        - y : array of M x 10 : y[i, j] = data sample i is number j ?
    """
    data = pandas.read_csv(TRAIN_FILE, delimiter=',')
    M = data.shape[0]               # Number of pictures
    N = data.shape[1]               # 785 = 28 * 28 pixels + 1
    y = data['label']               # Number written in the picture
    X = data.iloc[0:NB_SAMPLES,1:]  # features
    return M, N-1, X.values, y[0:NB_SAMPLES].values

def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data


##########################
# X-Validation
##########################
def findBestParams_RandomForest(X, y, X_test):
    pca = PCA()
    rf = RandomForestClassifier()
    pipeline = Pipeline([('pca', pca), ('rf', rf)])

    parameters = {
        'pca__n_components' : list(range(0.80,1.0,0.1)),
        'rf__n_estimators'  : [5,10,15,20,50,75,100],
        'rf__max_leaf_nodes': list(range(100,500,25))
    }

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=2)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    return estimator.best_estimator_.transform( X_test )

def findBestParams_LDA(X, y, X_test):
    lda = LDA()

    parameters = [{
        'solver'        : ['svd'],
        'n_components'  : list(range(20,400,10)) + [400, 450, 500, 550, 600, 650, 700]
    },
    {
        'solver'        : ['lsqr', 'eigen'],
        'n_components'  : list(range(20,400,10)) + [400, 450, 500, 550, 600, 650, 700],
        'shrinkage'     : list(range(0.1,1,0.1)) + [None, 'auto']
    }]

    estimator = GridSearchCV(lda, parameters, cv=10, n_jobs=2)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    return estimator.best_estimator_.transform( X_test )

def findBestParams_RegLog(X, y, X_test):
    pca = PCA()
    regLog = LogisticRegression(max_iter=10000)
    pipeline = Pipeline([('pca', pca), ('reglog', regLog)])

    parameters = [{
        'pca__n_components' : list(range(0.80,1.0,0.1)),
        'reglog__penalty'   : ['l1', 'l2'],
        'reglog__C'         : [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 10, 100],
        'reglog__solver'    : ['liblinear']

    },
    {
        'pca__n_components' : list(range(0.80,1.0,0.1)),
        'reglog__penalty'   : ['l2'],
        'reglog__C'         : [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 10, 100],
        'reglog__solver'    : ['newton-cg']
    },
    {
        'pca__n_components' : list(range(0.80,1.0,0.1)),
        'reglog__penalty'   : ['l2'],
        'reglog__C'         : [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 10, 100],
        'reglog__solver'    : ['lbfgs'],
        'reglog__multi_class' : ['ovr', 'multinomial']
    }]

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=2)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    return estimator.best_estimator_.transform( X_test )


##########################
def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)

    # ==== Cross validation & estimation ====
    y_hat_randomForest  = findBestParams_RandomForest(X_raw, y, X_test)
    y_hat_lda           = findBestParams_LDA(X_raw, y, X_test)
    y_hat_regLog        = findBestParams_RegLog(X_raw, y, X_test)

    return


#######################


if __name__ == "__main__":
    main()

