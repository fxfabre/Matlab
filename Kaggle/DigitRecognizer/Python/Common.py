#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas     # for read_csv function
import numpy as np
import scipy
import time
import sys
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

import xgboost as xgb


TRAIN_FILE = "../data/train.csv"
TEST_FILE  = "../data/test.csv"
NB_SAMPLES = 1000 # 42000
NB_THREADS = 1
VERBOSE_LEVEL = 3
NB_CV = 4

PCA_VARIANCES = [0.90] # , 0.80, 0.85, 0.90, 0.93, 0.96, 0.99]
REGLOG_C = [0.5, 1, 10] # [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]
SVM_C    = [0.5, 1, 10] # [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]



##########################
# Compute error
##########################
def computeError(y, y_hat):
    assert(len(y) == len(y_hat))
    diff = np.array(y) - np.array(y_hat)
    tmp = [ int(x==0) for x in diff ]
    return sum(tmp) / len(y)

def classifyComputeError(estimator, X, y):
    print( estimator)
    print( X )
    print( y )

    y_hat = estimator.predict( X )
    return computeError( y, y_hat )

##########################
# Dimension reduction
##########################
def pcaReduction(X, variance:float) -> PCA:
    # PCA('mle')   -> keep min(dataSamples, features) features
    # PCA( float ) -> keep the given variance
    # PCA( n )     -> keep n features
    pca = PCA(variance)
    X_afterPca = pca.fit_transform( X )

    nComponents = str(pca.n_components_)
    eigenValues = pca.explained_variance_ratio_
    explainedVariance = sum(eigenValues)

    print( 'Number of features to keep : ' + nComponents )
    print( 'PCA : Explained variance : '   + str(explainedVariance) )
    print( 'Variance of last component : ' + str(eigenValues[-1]) )

    return pca, X_afterPca, nComponents




