#!/usr/bin/python3
# -*- coding: utf-8 -*-

from Common import *
from Constants_best import *

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

#import xgboost as xgb


#############################

def classifyData(classifier, X_train, y, X_test, classifierName, variance=0):
    """
    :param classifier: classifier.fit and classifier.predict must be defined
    :param X_train:    training set
    :param y:          training target / label
    :param classifierName:  write test estimates to file classifierName.csv
    :return: predictions for X_test + % of good classification on train set
    """
    pca = None
    if variance != 0:
        classifierName += '_pca_' + str(variance)
        pca, X_train = pcaReduction(X_train, variance)
        X_test = pca.transform( X_test )

    classifier.fit(X_train, y)
    y_hat_train = classifier.predict(X_train)
    errorTrain = accuracy_score(y, y_hat_train)
    print( 'Train error {0} : {1}'.format(classifierName, str(errorTrain)) )

    y_hat = classifier.predict( X_test )
    lineNumber = list( range(1,len(y_hat)+1) )
    np.savetxt( 'Prediction/' + classifierName + '.txt', X=[lineNumber, y_hat], delimiter=',' )

    return y_hat

def knnClassif(X, y, X_test):
    knn = KNeighborsClassifier(**KNN_PARAM)
    y_hat = classifyData(knn, X, y, X_test, "KNN", 0.8)
    return y_hat

def ldaClassif(X, y, X_test):
    lda = LDA(**LDA_PARAMS)
    y_hat = classifyData(lda, X, y, X_test, "LDA", 0.8)
    return y_hat

def RandomForestClassif(X, y, X_test):
    rf = RandomForestClassifier(**RANDOM_FOREST_PARAM)
    y_hat = classifyData(rf, X, y, X_test, "RandomForest", 0.8)
    return y_hat

def svmClassif(X_train, y_train, X_test):
    svm = SVC( **SVM_PARAM )
    errorTrain = classifyData(svm, X_train, y_train, X_test, "SVM", 0.8)
    print( 'error SVM : ' + str(errorTrain) )

    y_hat = svm.predict(X_test)

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
    return
