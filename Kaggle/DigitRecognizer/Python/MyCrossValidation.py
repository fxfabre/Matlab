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

def findBestParams_all(X, y, classifierName, classifier, parameters):
    print( "\nCross validating {0}, pre-processing = PCA".format(classifierName))
#    findBestParams_PCA(X, y, PCA_VARIANCES, classifierName, classifier, parameters)

    print( "\nCross validating {0}, pre-processing = RBM".format(classifierName))
    findBestParams_RBM(X, y, RBM_N_COMPONENTS, RBM_LEARNING_RATE, classifierName, classifier, parameters)

    print( "\nCross validating {0}, pre-processing = ICA".format(classifierName))
    findBestParams_ICA(X, y, ICA_N_COMPONENTS, classifierName, classifier, parameters)

    print( "\nCross validating {0}, pre-processing = kernel PCA".format(classifierName))
    findBestParams_kernelPCA(X, y, PCA_VARIANCES, classifierName, classifier, parameters)

    print( "\nCross validating {0}, pre-processing = Projected grad NMF".format(classifierName))
    findBestParams_projectedGrad(X, y, GRAD_NMF_N_COMPONENTS, classifierName, classifier, parameters)

def findBestParams_None(X, y, classifierName, classifier, parameters):
    values = set( y )

    with open('Results/' + classifierName + '.log', 'w', 1) as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, no pre-processing".format(classifierName))

        findBestParams( X, y, values, classifierName, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_PCA(X, y, pcaValues, classifierName, classifier, parameters):
    values = set( y )
    classifierName = "PCA_" + classifierName

    with open('Results/' + classifierName + '.log', 'a') as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, pre-processing = PCA".format(classifierName))

        for variance in pcaValues:
            print("\nPCA, variance = " + str(variance))
            pca = PCA( variance )
            X_PCA = pca.fit_transform( X )
            name = '_'.join([classifierName, str(variance)])
            findBestParams( X_PCA, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_RBM(X, y, n_components, learningRates, classifierName, classifier, parameters):
    values = set( y )
    classifierName = "RBM_" + classifierName

    with open('Results/' + classifierName + '.log', 'a') as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, pre-processing = RBM".format(classifierName))

        for n_hiddenUnit in n_components:
            for learningRate in learningRates:
                print("\nRBM, {0} hidden unit, learning rate = {1}".format(n_hiddenUnit, learningRate) )

                rbm = BernoulliRBM( n_components=n_hiddenUnit, learning_rate=learningRate)
                X_RBM = rbm.fit_transform( X, y )
                name = '_'.join([classifierName, str(n_hiddenUnit), str(learningRate)])
                findBestParams( X_RBM, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_ICA(X, y, icaValues, classifierName, classifier, parameters):
    values = set( y )
    classifierName = "ICA_" + classifierName

    with open('Results/' + classifierName + '.log', 'a', 1) as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, pre-processing = ICA".format(classifierName))

        for n_params in icaValues:
            print("\nICA, {0} components".format(n_params) )

            ica = FastICA( n_params )
            X_ICA = ica.fit_transform( X )
            name = '_'.join([classifierName, str(n_params)])
            findBestParams( X_ICA, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams_kernelPCA(X, y, pcaValues, classifierName, classifier, parameters):
    values = set( y )
    classifierName = "kernelPCA_" + classifierName

    with open('Results/' + classifierName + '.log', 'a') as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, pre-processing = kernel PCA".format(classifierName))

        for n_params in pcaValues:
            print("\nKernel PCA, {0} parameters".format(n_params))
            pca = PCA( n_params )
            X_PCA = pca.fit_transform( X )
            name = '_'.join([classifierName, str(n_params)])
            findBestParams( X_PCA, y, values, name, classifier, parameters)
        sys.stdout = sys.__stdout__

def findBestParams_projectedGrad(X, y, n_components, classifierName, classifier, parameters):
    values = set( y )
    classifierName = "projectedGrad_" + classifierName
    X_positive = X - X.min(axis=0) # add to each column its min

    with open('Results/' + classifierName + '.log', 'a') as outFile:
        sys.stdout = outFile
        print( "Cross validating {0}, pre-processing = Projected grad NMF".format(classifierName))

        for n_params in n_components:
            print("\nProjected grad, {0} components".format(n_params) )

            gradNMF = ProjectedGradientNMF( n_params )
            X_gracNMF = gradNMF.fit_transform( X_positive, y )
            name = '_'.join([classifierName, str(n_params)])
            findBestParams( X_gracNMF, y, values, name, classifier, parameters )
        sys.stdout = sys.__stdout__

def findBestParams(X, y, y_values, classifierName, classifier, parameters):
    if len( y_values ) == 2:
        _findBestParams(X, y, classifierName, classifier, parameters)
        return

    for i in set(y):
        y_binaryClassif = [ int(x==i) for x in y ]
        _findBestParams( X, y_binaryClassif, classifierName + '_' + str(i), classifier, parameters )

def _findBestParams(X, y, classifierName, classifier, parameters):
    """ Run CrossValidation with 2 classes """

    estimator = GridSearchCV(classifier, parameters,
        cv=NB_CV, n_jobs=NB_THREADS, verbose=VERBOSE_LEVEL, scoring=classifyComputeError)
    estimator.fit(X, y)

    # Save best params and each CV score to a file
    if (estimator.best_score_ > 0.97):
        with open('Results/' + classifierName + '.bestParams', 'a') as outFile:
            outFile.write( str(estimator.best_params_) )
        with open('Results/' + classifierName + '.gridScores', 'a') as outFile:
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

