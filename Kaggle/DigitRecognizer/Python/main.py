#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas     # for read_csv function
import numpy as np
import scipy
import time
import sys
from datetime import datetime

from Common import *
from MyCrossValidation import *
from MyClassification import *
import Filters

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LassoCV

import matplotlib
from matplotlib import pyplot

## A regarder
# ridge regression
# Manifold learning
# Naive bayes ?
# LassoCV
# ElasticNetCV

# accuaracy_score, parametre normalize ?


##########################
# X-Validation, common models V2
##########################


# ElasticNetCV
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
    return M, N-1, X.values.astype(np.float64), y[0:NB_SAMPLES-1].values.astype(np.float64)

def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data

def findBestParams_KNN_2(X, y):
    knn = KNeighborsClassifier()
    parameters = {
        'n_neighbors'  : [3], # list(range(1,3,2)),
        'weights'      : ['uniform', 'distance'],
        'algorithm'    : ['kd_tree'], # ['ball_tree', 'kd_tree'],
        'leaf_size'    : [100], # [15, 30, 50, 100],
        'metric'       : ['minkowski'],
        'p'            : [1, 2]
    }

    for variance in PCA_VARIANCES:
        pca = PCA(variance)
        X_afterPca = pca.fit_transform( X )

        nComponents = str(pca.n_components_)
        eigenValues = pca.explained_variance_ratio_
        explainedVariance = sum(eigenValues)
        print( "%d components => %d variance".format(nComponents, explainedVariance) )

        findBestParams(X_afterPca, y, 'Result_KNN', knn, parameters)

    return


##########################
def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
#    (m_test, n_test, X_test) = readTestSet()
#    assert(n_raw == n_test)

    # === Pre processing ===
    scaler = preprocessing.StandardScaler().fit( X_raw )
    X_scaled = scaler.transform( X_raw )
    # mean(X_scaled), std(X_scaled) = 0, 1

    # X_scaled not invertible => remove columns
    pca, X_after_PCA = Filters.pca100pourcent( X_scaled )

    # === Feature selection : recherche dépendances linéaires.

    # === Feature selection : Z-Score
#    X_zscore, zScores = Filters.compute_Z_score(X_scaled, y)


    # === Feature selection : Lasso
    print( "Running Lasso")
    lassoCV = LassoCV(cv=5, n_jobs=2, max_iter=2000)
    lassoCV.fit(X_after_PCA, y)
    y_hat = lassoCV.predict( X_after_PCA )
    error = accuracy_score(y, y_hat, normalize=True)
    print( "Erreur avec LassoCV : " + str(error) )

    # ==== Cross validation & estimation ====
#    findBestParams_LDA(X_after_PCA, y)
#    findBestParams_RegLog(X_after_PCA, y)
#    findBestParams_KNN(X_after_PCA, y)
#    findBestParams_RandomForest(X_after_PCA, y)
#    findBestParams_SVM(X_after_PCA, y)
#    findBestParams_RBM(X_after_PCA, y)

#    knnClassif(X_raw, y)
#    svmClassif(X_raw, y)

#    XgradientBoost(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

