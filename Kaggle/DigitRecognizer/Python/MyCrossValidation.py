#!/usr/bin/python3
# -*- coding: utf-8 -*-


from Common import *
from datetime import datetime
import sys

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

import xgboost as xgb



##########################
# Tools
##########################

def findBestParams_PCA(X, y, pcaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "PCA_" + fileName

    for variance in pcaValues:
        pca = PCA( variance )
        X_after_PCA = pca.fit_transform( X )

        if len( values ) == 2:
            print( "PCA Cross validation, variance " + str(variance))
            _findBestParams(X_after_PCA, y, fileName + '_' + variance, classifier, parameters)
            return

        for i in set(y):
            print( "PCA Cross validation, class {0}, variance {1}".format(i, variance))
            y_binaryClassif = [ int(x==i) for x in y ]
            _findBestParams( X_after_PCA, y_binaryClassif, "{0}_{1}_{2}".format(fileName, i, variance), classifier, parameters )

def findBestParams_RBM(X, y, n_components, fileName, classifier, parameters):
    values = set( y )
    fileName = "RBM_" + fileName

    for n_hiddenUnit in n_components:
        rbm = BernoulliRBM( n_components=n_hiddenUnit)
        X_RBM = rbm.fit_transform( X, y )

        if len( values ) == 2:
            print( fileName + " Cross validation, variance " + str(n_hiddenUnit))
            _findBestParams(X_RBM, y, fileName + '_' + n_hiddenUnit, classifier, parameters)
            return

        for i in set(y):
            print( fileName + " Cross validation, class {0}, variance {1}".format(i, n_hiddenUnit))
            y_binaryClassif = [ int(x==i) for x in y ]
            _findBestParams( X_RBM, y_binaryClassif, "{0}_{1}_{2}".format(fileName, i, n_hiddenUnit), classifier, parameters )

def findBestParams_ICA(X, y, icaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "ICA_" + fileName

    for n_params in icaValues:
        ica = FastICA( n_params )
        X_after_ICA = ica.fit_transform( X )

        if len( values ) == 2:
            print( fileName + " Cross validation, N_params " + str(n_params))
            _findBestParams(X_after_ICA, y, fileName + '_' + n_params, classifier, parameters)
            return

        for i in set(y):
            print( fileName + " Cross validation, class {0}, N_params {1}".format(i, n_params))
            y_binaryClassif = [ int(x==i) for x in y ]
            _findBestParams( X_after_ICA, y_binaryClassif, fileName + '_' + str(i) + '_' + n_params, classifier, parameters )

def findBestParams_kernelPCA(X, y, pcaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "kernelPCA_" + fileName

    for n_params in pcaValues:
        pca = PCA( n_params )
        X_after_PCA = pca.fit_transform( X )

        if len( values ) == 2:
            print( fileName + " Cross validation, n_params " + str(n_params))
            _findBestParams(X_after_PCA, y, fileName + '_' + n_params, classifier, parameters)
            return

        for i in set(y):
            print( fileName + " Cross validation, class {0}, n_params {1}".format(i, n_params))
            y_binaryClassif = [ int(x==i) for x in y ]
            _findBestParams( X_after_PCA, y_binaryClassif, fileName + '_' + str(i) + '_' + n_params, classifier, parameters )

def findBestParams_projectedGrad(X, y, n_components, fileName, classifier, parameters):
    values = set( y )
    fileName = "projectedGrad_" + fileName

    for n_params in n_components:
        pca = PCA( n_params )
        X_after_PCA = pca.fit_transform( X )

        if len( values ) == 2:
            print( fileName + " Cross validation, n_params " + str(n_params))
            _findBestParams(X_after_PCA, y, fileName + '_' + n_params, classifier, parameters)
            return

        for i in set(y):
            print( fileName + " Cross validation, class {0}, n_params {1}".format(i, n_params))
            y_binaryClassif = [ int(x==i) for x in y ]
            _findBestParams( X_after_PCA, y_binaryClassif, fileName + '_' + str(i) + '_' + n_params, classifier, parameters )

def findBestParams(X, y, fileName, classifier, parameters):
    values = set( y )
    if len( values ) == 2:
        return _findBestParams(X, y, fileName, classifier, parameters)

    for i in set(y):
        y_binaryClassif = [ int(x==i) for x in y ]
        _findBestParams( X, y_binaryClassif, fileName + '_' + str(i), classifier, parameters )

# Run CrossValidation with 2 classes
def _findBestParams(X, y, fileName, classifier, parameters):
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
    regLog = LogisticRegression(max_iter=10000)
    parameters = [{
        'penalty'   : ['l1', 'l2'],
        'C'         : REGLOG_C,
        'solver'    : ['liblinear']
    },
    {
        'penalty'   : ['l2'],
        'C'         : REGLOG_C,
        'solver'    : ['newton-cg']
    },
    {
        'penalty'   : ['l2'],
        'C'         : REGLOG_C,
        'solver'    : ['lbfgs'],
        'multi_class' : ['ovr', 'multinomial']
    }]
    findBestParams_PCA(X, y, PCA_VARIANCES, 'Result_RegLog', regLog, parameters)
    return

def findBestParams_KNN(X, y):
    knn = KNeighborsClassifier()
    parameters = {
        'n_neighbors'  : [7, 11, 15, 19],
        'weights'      : ['uniform', 'distance'],
        'algorithm'    : ['kd_tree'], # 'ball_tree',
        'leaf_size'    : [100], # [15, 30, 50, 100],
        'metric'       : ['minkowski'],
        'p'            : [1, 2, np.inf]
    }
    findBestParams_PCA(X, y, PCA_VARIANCES, 'KNN', knn, parameters)
    return

def findBestParams_RandomForest(X, y):
    rf = RandomForestClassifier()
    parameters = {
        'n_estimators'  : [5, 10], # [5,10,20,50,75,100],
        'max_leaf_nodes': [100], #list(range(100,500,50))
    }
    findBestParams_PCA(X, y, PCA_VARIANCES, 'Result_RandomForest', rf, parameters)
    return

def findBestParams_SVM(X, y):
    svm = SVC()
    parameters = [{
        'C'            : SVM_C,
        'kernel'       : ['linear']
    },
    {
        'C'            : SVM_C,
        'kernel'       : ['poly'],
        'degree'       : [1, 2, 3, 4],
        'gamma'        : [1] # [0.001, 0.01, 0.1, 0.5, 1]
    },
    {
        'C'            : SVM_C,
        'kernel'       : ['rbf', 'sigmoid']
    }]
    findBestParams_PCA(X, y, PCA_VARIANCES, 'Result_SVM', svm, parameters)
    return

