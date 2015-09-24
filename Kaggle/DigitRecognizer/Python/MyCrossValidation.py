#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

from Common import *
from Constants_test import *
# from Constants import *

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.decomposition import PCA, FastICA, ProjectedGradientNMF
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score



###
#   FastICA did not converge. You might want to increase the number of iterations.
###s

##########################
# Tools
##########################

def findBestParams_all(X, y, fileName, classifier, parameters):
    print( "Cross validating {0}, pre-processing = PCA".format(fileName))
    findBestParams_PCA(X, y, PCA_VARIANCES, fileName, classifier, parameters)

    print( "Cross validating {0}, pre-processing = RBM".format(fileName))
    findBestParams_RBM(X, y, RBM_N_COMPONENTS, RBM_LEARNING_RATE, fileName, classifier, parameters)

    print( "Cross validating {0}, pre-processing = ICA".format(fileName))
    findBestParams_ICA(X, y, ICA_N_COMPONENTS, fileName, classifier, parameters)

    print( "Cross validating {0}, pre-processing = kernel PCA".format(fileName))
    findBestParams_kernelPCA(X, y, PCA_VARIANCES, fileName, classifier, parameters)

    print( "Cross validating {0}, pre-processing = Projected grad NMF".format(fileName))
    findBestParams_projectedGrad(X, y, GRAD_NMF_N_COMPONENTS, fileName, classifier, parameters)

def findBestParams_None(X, y, fileName, classifier, parameters):
    values = set( y )

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:
        sys.stdout = outFile

        findBestParams( X, y, values, fileName, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_PCA(X, y, pcaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "PCA_" + fileName

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile

        for variance in pcaValues:
            pca = PCA( variance )
            X_PCA = pca.fit_transform( X )
            name = '_'.join([fileName, str(variance)])
            findBestParams( X_PCA, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_RBM(X, y, n_components, learningRates, fileName, classifier, parameters):
    values = set( y )
    fileName = "RBM_" + fileName

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile

        for n_hiddenUnit in n_components:
            for learningRate in learningRates:
                rbm = BernoulliRBM( n_components=n_hiddenUnit, learning_rate=learningRate)
                X_RBM = rbm.fit_transform( X, y )
                name = '_'.join([fileName, str(n_hiddenUnit), str(learningRate)])
                findBestParams( X_RBM, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_ICA(X, y, icaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "ICA_" + fileName

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile

        for n_params in icaValues:
            ica = FastICA( n_params )
            X_ICA = ica.fit_transform( X )
            name = '_'.join([fileName, str(n_params)])
            findBestParams( X_ICA, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_kernelPCA(X, y, pcaValues, fileName, classifier, parameters):
    values = set( y )
    fileName = "kernelPCA_" + fileName

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile

        for n_params in pcaValues:
            pca = PCA( n_params )
            X_PCA = pca.fit_transform( X )
            name = '_'.join([fileName, str(n_params)])
            findBestParams( X_PCA, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_projectedGrad(X, y, n_components, fileName, classifier, parameters):
    values = set( y )
    fileName = "projectedGrad_" + fileName
    X_positive = X - X.min(axis=0) # add to each column its min

    with open('Results/' + fileName + '.log', 'w', 1) as outFile:     # 0 for unbuffered file
        sys.stdout = outFile

        for n_params in n_components:
            gradNMF = ProjectedGradientNMF( n_params )
            X_gracNMF = gradNMF.fit_transform( X_positive, y )
            name = '_'.join([fileName, str(n_params)])
            findBestParams( X_gracNMF, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams(X, y, y_values, fileName, classifier, parameters):
    if len( y_values ) == 2:
        _findBestParams(X, y, fileName, classifier, parameters)
        return

    for i in set(y):
        y_binaryClassif = [ int(x==i) for x in y ]
        _findBestParams( X, y_binaryClassif, fileName + '_' + str(i), classifier, parameters )

def _findBestParams(X, y, fileName, classifier, parameters):
    """ Run CrossValidation with 2 classes """

    estimator = GridSearchCV(classifier, parameters,
        cv=NB_CV, n_jobs=NB_THREADS, verbose=VERBOSE_LEVEL, scoring=classifyComputeError)
    estimator.fit(X, y)

    # Save best params and each CV score to a file
    with open('Results/ ' + fileName + '.bestParams', 'a') as outFile:
        outFile.write( str(estimator.best_params_) )
    with open('Results/ ' + fileName + '.gridScores', 'a') as outFile:
        outFile.write( str(estimator.grid_scores_) )

    return


##########################
# X-Validation, common models
##########################
def findBestParams_LDA(X, y):
    ## Cross validation
    findBestParams_None(X, y, 'LDA', LDA(), LDA_PARAMS)
    return

def findBestParams_RegLog(X, y):
    regLog = LogisticRegression(max_iter=10000)
    findBestParams_all(X, y, 'RegLog', regLog, REGLOG_PARAM)
    return

def findBestParams_KNN(X, y):
    knn = KNeighborsClassifier()
    findBestParams_all(X, y, 'KNN', knn, KNN_PARAM)
    return

def findBestParams_RandomForest(X, y):
    rf = RandomForestClassifier()
    findBestParams_all(X, y, 'RandomForest', rf, RANDOM_FOREST_PARAM)
    return

def findBestParams_SVM(X, y):
    svm = SVC()
    findBestParams_all(X, y, 'SVM', svm, SVM_PARAM)
    return

