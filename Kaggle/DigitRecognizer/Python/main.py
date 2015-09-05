#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas     # for read_csv function
import numpy as np
import scipy
import time
import sys
from datetime import datetime
from math import sqrt

from Common import *
from MyCrossValidation import *
from MyClassification import *

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from sklearn.linear_model import LassoCV

import matplotlib
from matplotlib import pyplot

## A regarder
# ridge regression
# Manifold learning
# Naive bayes ?
# LassoCV
# ElasticNetCV


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
    pca = PCA(784)
    pca.fit( X_scaled )
    eigenVal = [ x for x in pca.explained_variance_ if abs(x) > 1e-8 ]
    print("We can keep {0} components without loosing information".format(len(eigenVal)) )

    # Keep same information in dataset, remove useless parameters (parameters with variance < 1e-8)
    N_features = len(eigenVal)
    pca = PCA( N_features )
    X_after_PCA = pca.fit_transform( X_scaled )
    # if keep 640 components : det( X_after_PCA ) = 2.17628211139e-114
    # if keep 635 components : det( X_after_PCA ) = 1.42486552624e+22
    # if keep 630 components : det( X_after_PCA ) = 3.59056547778e+157
    # if keep 590 components : det( X_after_PCA ) = np.inf, variance > 0.99
    print("mean(X) = {0}, std(X) = {1}".format(
        max(np.mean(X_after_PCA, axis=0)),
        max(np.std( X_after_PCA, axis=0))
    ))


    # === Feature selection : recherche dépendances linéaires.
    XtX = np.dot( X_after_PCA.T, X_after_PCA )
    det = np.linalg.det( XtX )
    print("det( XtX ) = " + str(det))
    assert(det > 1e-3)
    # assert det not null, ie : matrix XtX not singular
    # assert that columns of X_after_PCA are not linearly dependants

    # === Feature selection : compute Z-score
    beta_hat = np.linalg.inv( XtX ).dot( X_after_PCA.T).dot(y)
    print( beta_hat )

    zScores = np.zeros( N_features )
    diago = np.diag( XtX )
    assert(len(diago) == N_features)
    sigma = np.std(X_after_PCA, axis=0)
    assert(len(diago) == len(sigma))
    for i in range( N_features ):
        zScores[i] = beta_hat[i] / (sigma[i] * sqrt(diago[i]))
    print("Z-scores : " + str(len(zScores)))
    print( zScores )

    scores_to_keep = [x for x in zScores if abs(x) > 2]
    print("Scores to keep : " + str(len(scores_to_keep)))
    print(scores_to_keep)

    return




    # === Feature selection : Lasso
    lassoCV = LassoCV(cv=5, n_jobs=2)
    lassoCV.fit(X_after_PCA)
    y_hat = lassoCV.predict( X_after_PCA )
    error = accuracy_score(y, y_hat, normalize=True)
    print( "Erreur avec LassoCV : " + str(error))

    # ==== Cross validation & estimation ====
#    findBestParams_LDA(X_raw, y)
#    findBestParams_RegLog(X_raw, y)
#    findBestParams_KNN(X_raw, y)
#    findBestParams_RandomForest(X_raw, y)
#    findBestParams_SVM(X_raw, y)
#    findBestParams_RBM(X_raw, y)

#    knnClassif(X_raw, y)
#    svmClassif(X_raw, y)

#    XgradientBoost(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

