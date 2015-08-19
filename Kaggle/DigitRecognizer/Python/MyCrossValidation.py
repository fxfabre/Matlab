#!/usr/bin/python3
# -*- coding: utf-8 -*-


from Common import *

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



##########################
# Tools
##########################
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
        'knn__algorithm'    : ['kd_tree'], # 'ball_tree',
        'knn__leaf_size'    : [100], # [15, 30, 50, 100],
        'knn__metric'       : ['minkowski'],
        'knn__p'            : [1, np.inf]
    }
    findBestParams(X, y, 'KNN', pipeline, parameters)
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

