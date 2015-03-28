#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas     # for read_csv function
import numpy
import time

from sklearn.decomposition import PCA

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


TRAIN_FILE = "../data/train.csv"
TEST_FILE  = "../data/test.csv"
NB_SAMPLES = 5000

##########################
# Input data
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
    X = data.iloc[0:NB_SAMPLES,1:]  # features
    return M, N-1, X, y[0:NB_SAMPLES]
def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data
def saveArrayToFile(y:numpy.ndarray, fileName):
    fullName = "{0}_{1}.csv".format(time.strftime('%Y%m%d_%H%M%S'), fileName)
    numpy.savetxt(fullName, y, fmt="%d", delimiter=",")
    print("file " + fullName + " saved")
def savePredictionsToFile(y:pandas.core.series.Series, fileName):
    fullName = "{0}_{1}.csv".format(time.strftime('%Y%m%d_%H%M%S'), fileName)
    y.to_csv(fullName, False)
    print("file " + fullName + " saved")


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

    print( 'Number of features to keep : ' + nComponents)
    print( 'PCA : Explained variance : ' + str(explainedVariance))

    return pca, X_afterPca


##########################
# Classification
##########################
def classifyData(classifier, X_train, y_train, X_test, y_test=None, classifierName=''):
    """
    :param classifier: classifier.fit and classifier.predict must be defined
    :param X_train:     training set
    :param y_train:     training target / label
    :param X_test:      test set                if not provided, use X_train and y_train
    :param y_test:      test target / label
    :param classifierName: if provided, write test estimates to file classifierName.csv
    :return: if y_test provided, return the % of good classification on test set
            otherwise, return error on training set
    """
    classifier.fit(X_train, y_train)
    error = 0

    if X_test is None:
        X_test = X_train
        y_test = y_train

    if y_test is None:
        y_hat = classifier.predict(X_train)
        error = accuracy_score(y_train, y_hat)
    else:
        y_hat = classifier.predict(X_test)
        error = accuracy_score(y_test, y_hat)

    if len(classifierName) > 0:
        saveArrayToFile(y_hat, classifierName)

    return error

def linearDiscriminantAnalysis(X_train, y_train, X_test=None, y_test=None):
    lda = LDA()
    error = classifyData(lda, X_train, y_train, X_test, y_test, 'LDA')
    return lda, error

def logisticReg(X_train, y_train, X_test=None, y_test=None):
    regLog = LogisticRegression()
    error = classifyData(regLog, X_train, y_train, X_test, y_test, 'RegLog')
    return regLog, error

def knnClassif(X_train, y_train, X_test=None, y_test=None):
    knn = KNeighborsClassifier()
    error = classifyData(knn, X_train, y_train, X_test, y_test, "KNN")
    return knn, error

def svmClassif(X_train, y_train, X_test=None, y_test=None):
    svm = SVC()
    error = classifyData(svm, X_train, y_train, X_test, y_test, "SVM")
    return svm, error

def randomForest(X_train, y_train, X_test=None, y_test=None):
    forest = RandomForestClassifier()
    error = classifyData(forest, X_train, y_train, X_test, y_test, 'RandomForest')
    return forest, error

def restrictedBoltzmanMachine(X_train, y_train, X_test=None, y_test=None):
    # Models we will use
    logistic = LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    # Training
    # Hyper-parameters. These were set by cross-validation, using a GridSearchCV.
    # Here we are not performing cross-validation to save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20

    # More components tend to give better prediction performance, but larger fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    error = classifyData(classifier, X_train, y_train, X_test, y_test, 'RBM')

    # Training Logistic regression
    regLog = LogisticRegression(C=100.0)
    regLog.fit(X_train, y_train)

    # Evaluation
    if y_test is None:
        X_test = X_train
        y_test = y_train

    print("Logistic regression using RBM features:\n%s\n" %
        classification_report(y_test, classifier.predict(X_test)) )

    print("Logistic regression using raw pixel features:\n%s\n" %
        classification_report(y_test, regLog.predict(X_test)))

    return classifier, error


##########################
def main():
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)

    # Variance = 0.90 => 85  features
    # Variance = 0.95 => 149 features
    # Variance = 0.96 => 174 features
    # Variance = 0.97 => 207 features
    # Variance = 0.98 => 252 features
    # Variance = 0.99 => 322 features
    (pca, X_afterPca) = pcaReduction(X_raw, 0.97)
    X_test_afterPca = pca.transform(X_test)

    (lda, error) = linearDiscriminantAnalysis(X_afterPca, y, X_test_afterPca)
    print( "LDA accuracy : " + str(error) )

    (regLog, error) = logisticReg(X_afterPca, y, X_test_afterPca)
    print( "RegLog accuracy : " + str(error) )

    (knn, error) = knnClassif(X_afterPca, y, X_test_afterPca)
    print( "KNN accuracy : " + str(error) )

    (svm, error) = svmClassif(X_afterPca, y, X_test_afterPca)
    print( "SVM accuracy : " + str(error) )

    (rbmClassif, error) = restrictedBoltzmanMachine(X_raw, y, X_test_afterPca)
    print( "RBM accuracy : " + str(error) )

if __name__ == "__main__":
    main()