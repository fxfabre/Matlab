#!/usr/bin/python3
# -*- coding: utf-8 -*-

from Common import *

import pandas     # for read_csv function
import numpy as np
import scipy
import time
import sys
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

import xgboost as xgb


#############################

def classifyData(classifier, X_train, y, X_test, classifierName, variance=1):
    """
    :param classifier: classifier.fit and classifier.predict must be defined
    :param X_train:    training set
    :param y:          training target / label
    :param classifierName:  write test estimates to file classifierName.csv
    :return: predictions for X_test + % of good classification on train set
    """
    pca = None
    if variance < 1:
        pca, X_train = pcaReduction(X_train, variance)
        X_test = pca.transform( X_test )

    classifier.fit(X_train, y)
    y_hat_train = classifier.predict(X_train)
    errorTrain = computeError(y, y_hat_train)
    print( 'Train error {0} : {1}'.format(classifierName, str(errorTrain)) )

    y_hat = classifier.predict( X_test )
    np.savetxt( 'Prediction/' + classifierName + '.txt', y_hat )

    return y_hat

def knnClassif(X, y, X_test):
    parameters = {
        'n_neighbors'  : 15,
        'weights'      : 'uniform',
        'algorithm'    : 'kd_tree',
        'leaf_size'    : 30,
        'metric'       : 'minkowski',
        'p'            : 1
    }
    knn = KNeighborsClassifier(**parameters)
    y_hat = classifyData(knn, X, y, X_test, "KNN", 0.8)
    return y_hat

def ldaClassif(X, y, X_test):
    parameters = {
        'solver'    : 'lsqr',
        'shrinkage' : 'auto'
    }
    lda = LDA(**parameters)
    y_hat = classifyData(lda, X, y, X_test, "LDA", 0.8)
    return y_hat

def RandomForestClassif(X, y, X_test):
    parameters = {
        'n_estimator'   : '100',
        'max_leaf_node' : '200'
    }
    rf = RandomForestClassifier(**parameters)
    y_hat = classifyData(rf, X, y, X_test, "RandomForest", 0.8)
    return y_hat

def svmClassif(X_train, y_train):
    pca = PCA(n_components=10)
    Xt = pca.fit_transform(X_train)

    svm = SVC(kernel='linear')
    errorTrain = classifyData(svm, Xt, y_train, "SVM")
    print( 'error SVM : ' + str(errorTrain) )
    return

##########################
# XGBoost
##########################
def XgradientBoost(X, y, X_test):
    y = (y == 1)
    y = [int(x) for x in y]

    dtrain = xgb.DMatrix( X, label=y)

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eta'] = 0.1
    param['max_depth'] = 7
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['min_child_weight'] = 100
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['nthread'] = NB_THREADS

    num_round = 50
    #xgb.cv(param, dtrain, num_round, nfold=5)
    model = xgb.train(param, dtrain, num_round)
    print( model )

    # dump model
    model.dump_model('dump.raw.txt')
    # dump model with feature map
    model.dump_model('dump.raw.txt', 'featmap.txt')

    y_hat = model.predict( X_test )

#    ypred = bst.predict(dtest)

