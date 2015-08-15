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
NB_SAMPLES = 5000 # 42000
NB_THREADS = 1
VERBOSE_LEVEL = 3
NB_CV = 4

PCA_VARIANCES = [0.90] # , 0.80, 0.85, 0.90, 0.93, 0.96, 0.99]
REGLOG_C = [0.5, 1, 10] # [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]
SVM_C    = [0.5, 1, 10] # [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]


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
# Tools
##########################
def findBestParams(X, y, fileName, classifier, parameters):
    print( str( datetime.now() ) + " : Processing of " + fileName)

    # Run cross validation and save displayed text in log file
    with open('Results/ ' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile
        estimator = GridSearchCV(classifier, parameters,
            cv=NB_CV, n_jobs=NB_THREADS, verbose=VERBOSE_LEVEL, scoring=classifyComputeError)
        estimator.fit(X, y)

    # Set default output for print to the screen (hope there is no print between those 2 lines)
    sys.stdout = sys.__stdout__

    # Save best params and each CV score to a file
    with open('Results/ ' + fileName + '.bestParams', 'w') as outFile:
        outFile.write( str(estimator.best_params_) )
    with open('Results/ ' + fileName + '.gridScores', 'w') as outFile:
        outFile.write( str(estimator.grid_scores_) )

    print( estimator.best_params_ )
    return

def optimizeParameters(classifier, X, y, n_iter, params={}, steps={}):
    for n in range( n_iter ):
        for key, value in params.items():
            paramsMoins = dict( params ) # clone params
            paramsMoins[key] = value - steps[key]


#############################
def classifyComputeError(estimator, X, y):
    print( estimator)
    print( X )
    print( y )

    y_hat = estimator.predict( X )
    return computeError( y, y_hat )

def computeError(y, y_hat):
    assert(len(y) == len(y_hat))
    diff = np.array(y) - np.array(y_hat)
    tmp = [ int(x==0) for x in diff ]
    return sum(tmp) / len(y)

def classifyData(classifier, X_train, y_train, classifierName=''):
    """
    :param classifier: classifier.fit and classifier.predict must be defined
    :param X_train:         training set
    :param y_train:         training target / label
    :param classifierName:  write test estimates to file classifierName.csv
    :return: predictions for X_test + % of good classification on train set
    """
    error_test = 0

    classifier.fit(X_train, y_train)
    y_hat_train = classifier.predict(X_train)
    errorTrain = computeError(y_train, y_hat_train)
    return errorTrain

def knnClassif(X_train, y_train):
    parameters = {
        'n_neighbors'  : 1,
        'weights'      : 'uniform',
        'algorithm'    : 'kd_tree',
        'leaf_size'    : 100,
        'metric'       : 'minkowski',
        'p'            : 1
    }
    knn = KNeighborsClassifier(**parameters)
    errorTrain = classifyData(knn, X_train, y_train, "KNN")
    print( 'error KNN : ' + str(errorTrain) )
    return

def svmClassif(X_train, y_train):
    pca = PCA(n_components=10)
    Xt = pca.fit_transform(X_train)

    svm = SVC(kernel='linear')
    errorTrain = classifyData(svm, Xt, y_train, "SVM")
    print( 'error SVM : ' + str(errorTrain) )
    return


##########################
# X-Validation, common models
##########################
def findBestParams_LDA(X, y):
    ## Cross validation
    parameters = [{
        'solver'        : ['svd'],
        'n_components'  : PCA_VARIANCES
    },
    {
        'solver'        : ['lsqr', 'eigen'],
        'n_components'  : PCA_VARIANCES,
        'shrinkage'     : [1] # list(np.arange(0.125, 1.1, 0.125)) + [None, 'auto']
    }]
    findBestParams(X, y, 'LDA', LDA(), parameters)
    return

def findBestParams_RegLog(X, y):
    pipeline = Pipeline([('pca', PCA()), ('reglog', LogisticRegression(max_iter=10000))])

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
    findBestParams(X, y, 'Result_RegLog', pipeline, parameters)
    return

def findBestParams_KNN(X, y):
    pipeline = Pipeline([('pca', PCA()), ('knn', KNeighborsClassifier())])
    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'knn__n_neighbors'  : [3], # list(range(1,3,2)),
        'knn__weights'      : ['uniform', 'distance'],
        'knn__algorithm'    : ['ball_tree', 'kd_tree'],
        'knn__leaf_size'    : [100], # [15, 30, 50, 100],
        'knn__metric'       : ['minkowski'],
        'knn__p'            : [1, 2]
    }
    findBestParams(X, y, 'Result_KNN', pipeline, parameters)
    return

def findBestParams_RandomForest(X, y):
    rf = RandomForestClassifier()
    pipeline = Pipeline([('pca', PCA()), ('rf', rf)])
    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'rf__n_estimators'  : [5, 10], # [5,10,20,50,75,100],
        'rf__max_leaf_nodes': [100], #list(range(100,500,50))
    }
    findBestParams(X, y, 'Result_RandomForest', pipeline, parameters)
    return

def findBestParams_SVM(X, y):
    pipeline = Pipeline([('pca', PCA()), ('svm', SVC())])

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
    findBestParams(X, y, 'Result_SVM', pipeline, parameters)
    return

def findBestParams_RBM(X, y):
    pca = PCA()
    rbm = BernoulliRBM()
    pipeline = Pipeline([('pca', pca), ('rbm', rbm)])

    parameters = {
        'pca__n_components' : PCA_VARIANCES,
        'rbm__n_components' : [50, 200, 500, 1000],
        'rbm__learning_rate': [0.1], # [0.001, 0.01, 0.1],
        'rbm__n_iter'       : ['liblinear']
    }
    findBestParams(X, y, 'Result_BernoulliRBM', pipeline, parameters)
    return


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

##########################
def main():
    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)

    # ==== Cross validation & estimation ====
#    findBestParams_LDA(X_raw, y)
#    findBestParams_RegLog(X_raw, y)
#    findBestParams_KNN_2(X_raw, y)
#    findBestParams_RandomForest(X_raw, y)
#    findBestParams_SVM(X_raw, y)
#    findBestParams_RBM(X_raw, y)

    knnClassif(X_raw, y)
    svmClassif(X_raw, y)

#    XgradientBoost(X_raw, y, X_test)

    return


#######################
if __name__ == "__main__":
    main()

