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
from LasagneNN2 import lasagneNN

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



## A regarder
# ridge regression / ridgeCV
# Manifold learning
# Naive bayes ?
# LassoCV
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
    print("read training set")
    data = pandas.read_csv(TRAIN_FILE, delimiter=',')
    M = data.shape[0]               # Number of pictures
    N = data.shape[1]               # 785 = 28 * 28 pixels + 1
    y = data['label']               # Number written in the picture
    X = data.iloc[1:NB_SAMPLES+1, 1:]  # features

    print( "Shape X : " + str(X.shape))
    return M, N-1, X.values, y[0:NB_SAMPLES], None

    # PCA : keep 99% of variance
    pca, X, N = pcaReduction(X.values, 0.99)
    X = np.array( X )
    print( "Shape X : " + str(X.shape))

    # Mean - center the matrix X
    print("Updating X to have mean(X) = 0, std(X) = 1")
    X = ( X - X.mean(0) )
    X = [ X[i,:] / X.std(0) for i in range(len(X)) ]
    X = np.array( X )

    print( "Shape X : " + str(X.shape))
    return M, N, X, y[0:NB_SAMPLES], pca

def readTestSet(pca=None):
    print("Reading test set")
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels

    if pca is None:
        X_test = data.values
    else:
        X_test = pca.transform(data.values)

    return M, N, X_test


##########################
def runCrossValidation(X_train, y_train):
    print("Starting cross validation")
#    findBestParams_LDA(X_train, y_train)
#    findBestParams_RegLog(X_train, y_train)
#    findBestParams_KNN(X_train, y_train)
#    findBestParams_RandomForest(X_train, y_train)
#    findBestParams_SVM(X_train, y_train)
#    findBestParams_RBM(X_train, y_train)
    return

def runClassifiation(X_train, y_train, X_test):
    print("Starting classification")
    ldaClassif(X_train, y_train, X_test)
    RandomForestClassif(X_train, y_train, X_test)
#    knnClassif(X_train, y_train, X_test)
#    svmClassif(X_train, y_train)

#    XgradientBoost(X_raw, y, X_test)
    return

def runNeuralNetworks(X_train, y_train, X_test):
    print("Starting GPU neural networks")

#    y = [ int(x==5) for x in y_train ]
    y_train = np.array(y_train)
    lasagneNN( X_train, y_train)

def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y, pca) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet(pca)
    assert(n_raw == n_test)


    runCrossValidation(X_raw, y)

    runClassifiation(X_raw, y, X_test)

    runNeuralNetworks(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

