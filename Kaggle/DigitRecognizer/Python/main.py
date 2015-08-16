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

from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

import xgboost as xgb


## A regarder
# ridge regression
# Manifold learning
# Naive bayes ?
# LassoCV
# ElasticNetCV


##########################
# X-Validation, common models V2
##########################
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
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)

    # ==== Cross validation & estimation ====
#    findBestParams_LDA(X_raw, y)
#    findBestParams_RegLog(X_raw, y)
    findBestParams_KNN(X_raw, y)
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

