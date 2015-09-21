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
    X = data.iloc[0:NB_SAMPLES, 1:]  # features

    # Normalize [0, 1]
    X /= 255.0
    X = X.values.astype(np.float64)
    y = y[0:NB_SAMPLES].values
    return M, N-1, X, y

def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data


##########################
def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    print( X_raw.shape )
    print( y.shape )

#    (m_test, n_test, X_test) = readTestSet()
#    assert(n_raw == n_test)

    # === Pre processing ===
    # X_raw not invertible => remove columns
    pca, X_after_PCA = Filters.pca100pourcent( X_raw )

    scaler = preprocessing.StandardScaler().fit( X_after_PCA )
    X_scaled = scaler.transform( X_after_PCA )
    # mean(X_scaled), std(X_scaled) = 0, 1

    # === Feature selection : recherche dépendances linéaires.

    # === Feature selection : Z-Score
#    X_zscore, zScores = Filters.compute_Z_score(X_raw, y)


    # === Feature selection : Lasso
#    print( "Running Lasso")
#    lassoCV = LassoCV(cv=5, n_jobs=2, max_iter=2000)
#    lassoCV.fit(X_after_PCA, y)
#    y_hat = lassoCV.predict( X_after_PCA )
#    error = accuracy_score(y, y_hat, normalize=True)
#    print( "Erreur avec LassoCV : " + str(error) )

    # Correlation matrix
#    correlation = X_scaled.T.dot( X_scaled ) / X_scaled.shape[0]
#    pyplot.matshow(correlation)
#    pyplot.show()


    # ==== Cross validation & estimation ====
    findBestParams_LDA(X_scaled, y)
    findBestParams_RegLog(X_scaled, y)
    findBestParams_KNN(X_scaled, y)
    findBestParams_RandomForest(X_scaled, y)
    findBestParams_SVM(X_scaled, y)

#    knnClassif(X_raw, y)
#    svmClassif(X_raw, y)

#    XgradientBoost(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

