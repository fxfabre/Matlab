#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas     # for read_csv function
import numpy as np
import time
import sys

from sklearn.decomposition import PCA

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

TRAIN_FILE = "../data/train.csv"
TEST_FILE  = "../data/test.csv"
NB_SAMPLES = 2000 # 42000
NB_THREADS = 2

PCA_VARIANCES = [0.75] # , 0.80, 0.85, 0.90, 0.93, 0.96, 0.99]
REGLOG_C = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]
SVM_C    = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]


## A regarder
# ridge regression
# Manifold learning
# Naive bayes ?


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
    X = data.iloc[1:NB_SAMPLES, :]  # features
    return M, N-1, X.values, y[0:NB_SAMPLES-1].values

def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data


##########################
# X-Validation
##########################
def findBestParams_LDA(X, y, X_test):
    file = open('Result_LDA', 'w')
    sys.stdout = file
    lda = LDA()

    parameters = [{
        'solver'        : ['svd'],
        'n_components'  : PCA_VARIANCES
    },
    {
        'solver'        : ['lsqr', 'eigen'],
        'n_components'  : PCA_VARIANCES,
        'shrinkage'     : [1] # list(np.arange(0.125, 1.1, 0.125)) + [None, 'auto']
    }]

    estimator = GridSearchCV(lda, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return

def findBestParams_RegLog(X, y, X_test):
    file = open('Result_RegLog', 'w')
    sys.stdout = file

    pca = PCA()
    regLog = LogisticRegression(max_iter=10000)
    pipeline = Pipeline([('pca', pca), ('reglog', regLog)])

    parameters = [{
        'pca__n_components' : PCA_VARIANCES,
        'reglog__penalty'   : ['l1', 'l2'],
        'reglog__C'         : REGLOG_C,
        'reglog__solver'    : ['liblinear']

    },
    {
        'pca__n_components' : PCA_VARIANCES,
        'reglog__penalty'   : ['l2'],
        'reglog__C'         : REGLOG_C,
        'reglog__solver'    : ['newton-cg']
    },
    {
        'pca__n_components' : PCA_VARIANCES,
        'reglog__penalty'   : ['l2'],
        'reglog__C'         : REGLOG_C,
        'reglog__solver'    : ['lbfgs'],
        'reglog__multi_class' : ['ovr', 'multinomial']
    }]

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return

def findBestParams_KNN(X, y, X_test):
    file = open('Result_KNN', 'w')
    sys.stdout = file

    pca = PCA()
    knn = KNeighborsClassifier()
    pipeline = Pipeline([('pca', pca), ('knn', knn)])

    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'knn__n_neighbors'  : list(range(1,3,2)),
        'knn__weights'      : ['uniform', 'distance'],
        'knn__algorithm'    : ['ball_tree', 'kd_tree'],
        'knn__leaf_size'    : [15, 30, 50, 100],
        'knn__metric'       : ['minkowski'],
        'knn__p'            : [1, 2]
    }

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return

def findBestParams_RandomForest(X, y, X_test):
    file = open('Result_RandomForest', 'w')
    sys.stdout = file

    pca = PCA()
    rf = RandomForestClassifier()
    pipeline = Pipeline([('pca', pca), ('rf', rf)])

    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'rf__n_estimators'  : [5, 10], # [5,10,20,50,75,100],
        'rf__max_leaf_nodes': [100], #list(range(100,500,50))
    }

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return

def findBestParams_SVM(X, y, X_test):
    file = open('Result_SVM', 'w')
    sys.stdout = file

    pca = PCA()
    svm = SVC()
    pipeline = Pipeline([('pca', pca), ('svm', svm)])

    parameters = [{
        'pca__n_components' : PCA_VARIANCES,
        'svm__C'            : SVM_C,
        'svm__kernel'       : ['linear']
    },
    {
        'pca__n_components' : PCA_VARIANCES,
        'svm__C'            : SVM_C,
        'svm__kernel'       : ['poly'],
        'svm__degree'       : [1, 2, 3, 4],
        'gamma'             : [1] # [0.001, 0.01, 0.1, 0.5, 1]
    },
    {
        'pca__n_components' : PCA_VARIANCES,
        'svm__C'            : SVM_C,
        'svm__kernel'       : ['rbf', 'sigmoid']
    }]

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return

def findBestParams_RBM(X, y, X_test):
    file = open('Result_BernoulliRBM', 'w')
    sys.stdout = file

    pca = PCA()
    rbm = BernoulliRBM()
    pipeline = Pipeline([('pca', pca), ('rbm', rbm)])

    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'rbm__n_components' : [50, 200, 500, 1000],
        'rbm__learning_rate': [0.001, 0.01, 0.1],
        'rbm__n_iter'       : ['liblinear']

    }

    estimator = GridSearchCV(pipeline, parameters, cv=10, n_jobs=NB_THREADS, verbose=3)
    estimator.fit(X, y)

    print( estimator.best_params_ )
    sys.stdout = sys.__stdout__
    file.close()
    return


##########################
def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)

    # ==== Cross validation & estimation ====
    findBestParams_LDA(X_raw, y, X_test)
    findBestParams_RegLog(X_raw, y, X_test)
    findBestParams_KNN(X_raw, y, X_test)
    findBestParams_RandomForest(X_raw, y, X_test)
    findBestParams_SVM(X_raw, y, X_test)
    findBestParams_RBM(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

