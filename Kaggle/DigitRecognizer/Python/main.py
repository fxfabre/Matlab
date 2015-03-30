#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas     # for read_csv function
import numpy
import time

from email.mime.multipart import MIMEMultipart
import smtplib

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
from sklearn.grid_search import GridSearchCV


TRAIN_FILE = "../data/train.csv"
TEST_FILE  = "../data/test.csv"
OUTPUT_FOLDER = "output"
RESULT_FILES = OUTPUT_FOLDER + "/Best_Classification.csv"
EMAIL_FILE = "emailAdress.txt"
NB_SAMPLES = 1000
VARIANCE_TO_KEEP = 0.85


##########################
# Tools
##########################
def computeError(y, y_hat):
    assert(len(y) == len(y_hat))

    nbGoodClassif = 0
    for i in range( len(y) ):
        if y[i] == y_hat[i]:
            nbGoodClassif += 1

    return nbGoodClassif / len(y)
    return accuracy_score(y, y_hat)

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
    X = data.iloc[0:NB_SAMPLES,1:]  # features
    return M, N-1, X, y[0:NB_SAMPLES]
def readTestSet():
    data = pandas.read_csv(TEST_FILE,delimiter=',')
    M = data.shape[0]   # Number of pictures
    N = data.shape[1]   # 784 = 28 * 28 pixels
    return M, N, data
def saveArrayToFile(y:numpy.ndarray, fileName):
    if NB_SAMPLES != 42000:
        return

    fullName = "{0}/{1}_{2}_{3}_{4}.csv".format(
        OUTPUT_FOLDER,
        time.strftime('%Y%m%d_%H%M%S'),
        VARIANCE_TO_KEEP,
        NB_SAMPLES,
        fileName
    )
    f = open(fullName, 'w')
    f.write("ImageId,Label\n")
    for i in range(len(y)):
        f.write("{0},{1}\n".format(i+1, y[i]))
    f.close()
#    numpy.savetxt(fullName, y, fmt="%d", delimiter=",")
    print("file " + fullName + " saved")
def readOutPutFiles():
    data = pandas.read_csv(RESULT_FILES, delimiter=',')
    return data

def sendEmail():
    msg = MIMEMultipart()
    msg['Subject'] = "Job done"
    msg['From'] = ""
    msg['To'] = ""

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()

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
def classifyData(classifier, X_train, y_train, X_test, classifierName=''):
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
    error_test = 0

    classifier.fit(X_train, y_train)
    y_hat_train = classifier.predict(X_train)
    errorTrain = computeError(y_train, y_hat_train)
    saveArrayToFile(y_hat_train, "Train_" + classifierName)

    y_hat_test  = classifier.predict(X_test)
    saveArrayToFile(y_hat_test, "Test_" + classifierName)

    return y_hat_test, errorTrain

    error = 0

    if X_test is None:
        X_test = X_train
        y_test = y_train

    if y_test is None:
        y_hat = classifier.predict(X_train)
        error = computeError(y_train, y_hat)
        y_hat = classifier.predict(X_test)
    else:
        y_hat = classifier.predict(X_test)
        error = computeError(y_test, y_hat)

    if len(classifierName) > 0:
        saveArrayToFile(y_hat, classifierName)

    return error

def linearDiscriminantAnalysis(X_train, y_train, X_test):
    lda = LDA()
    (y_hat_test, errorTrain) = classifyData(lda, X_train, y_train, X_test, 'LDA')
    return lda, y_hat_test, errorTrain

def logisticReg(X_train, y_train, X_test):
    regLog = LogisticRegression()
    (y_hat_test, errorTrain) = classifyData(regLog, X_train, y_train, X_test, 'RegLog')
    return regLog, y_hat_test, errorTrain

def knnClassif(X_train, y_train, X_test):
    knn = KNeighborsClassifier()
    (y_hat_test, errorTrain) = classifyData(knn, X_train, y_train, X_test, "KNN")
    return knn, y_hat_test, errorTrain

def svmClassif(X_train, y_train, X_test, kernel='linear'):
    svm = SVC(kernel=kernel)
    (y_hat_test, errorTrain) = classifyData(svm, X_train, y_train, X_test, "SVM")
    return svm, y_hat_test, errorTrain

def randomForest(X_train, y_train, X_test):
    forest = RandomForestClassifier()
    (y_hat_test, errorTrain) = classifyData(forest, X_train, y_train, X_test, 'RandomForest')
    return forest, y_hat_test, errorTrain

def restrictedBoltzmanMachine(X_train, y_train, X_test):
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
    rbm.n_components = 150
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    (y_hat_test, errorTrain) = classifyData(classifier, X_train, y_train, X_test, 'RBM')

    # Training Logistic regression
    regLog = LogisticRegression(C=100.0)
    regLog.fit(X_train, y_train)

    # Evaluation
#    if y_test is None:
#        X_test = X_train
#        y_test = y_train

#    print("Logistic regression using RBM features:\n%s\n" %
#        classification_report(y_test, classifier.predict(X_test)) )

#    print("Logistic regression using raw pixel features:\n%s\n" %
#        classification_report(y_test, regLog.predict(X_test)))

    return classifier, y_hat_test, errorTrain


##########################
# X-Validation
##########################
def crossValidateSvmGrid(X_train, y_train):
    parameters = {
        'kernel' : ['rbf'], #, 'linear', 'rbf', 'sigmoid'),
        'C'      : [0.4, 0.6, 0.8, 1.0],
        'gamma'  : [0.0]
    }
    svm = SVC()
    xValid = GridSearchCV(svm, parameters, n_jobs=2, pre_dispatch=10, cv=4)
    xValid.fit(X_train, y_train)
    return xValid.best_params_

def crossValidateSvmCustom(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    cValues = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]
    gammaValues = [0.0, 0.05, 0.1, 0.5]
    kernels = ['linear', 'rbf', 'sigmoid']

    for k in kernels:
        for gVal in gammaValues:
            for cVal in cValues:
                parameters = { 'kernel': k, 'C': cVal, 'gamma': gVal }
                svm = SVC(kernel=k, C=cVal, gamma=gVal)
                svm.fit( X_train, y_train)
                y_hat = svm.predict(X_test)

                error = computeError(y_test, y_hat)
                print( str(parameters) + " => " + str(error) )
    return

##########################
def main():
    error = 0

    # ==== READ DATA ====
    (m_raw, n_raw, X_raw, y) = readTrainingSet()
    (m_test, n_test, X_test) = readTestSet()
    assert(n_raw == n_test)


    # TODO : cross validate here each models
    # TODO ; check SVM and RBM, they does not work


    # ==== DIMENSION REDUCTION ====
    # Variance = 0.90 => 85  features
    # Variance = 0.95 => 149 features
    # Variance = 0.96 => 174 features
    # Variance = 0.97 => 207 features
    # Variance = 0.98 => 252 features
    # Variance = 0.99 => 322 features
    (pca, X_afterPca) = pcaReduction(X_raw, VARIANCE_TO_KEEP)
    X_test_afterPca = pca.transform(X_test)

    # ==== CROSS VALIDATION ====
#    crossValidateSvmCustom(X_afterPca, y)
#    bestParams = crossValidateSvmGrid(X_afterPca, y)
#    print( bestParams )

    # ==== CLASSIFICATION ====
    (lda, y_hat_lda, error) = linearDiscriminantAnalysis(X_afterPca, y, X_test_afterPca)
    print( "LDA accuracy : " + str(error) )

    (regLog, y_hat_regLog, error) = logisticReg(X_afterPca, y, X_test_afterPca)
    print( "RegLog accuracy : " + str(error) )

    (knn, y_hat_knn, error) = knnClassif(X_afterPca, y, X_test_afterPca)
    print( "KNN accuracy : " + str(error) )

    (forest, y_hat_forest, error) = randomForest(X_afterPca, y, X_test_afterPca)
    print( "Random forest accuracy : " + str(error) )

#    (svm, y_hat_svm, error) = svmClassif(X_afterPca, y, X_test_afterPca)
    print( "SVM accuracy : " + str(error) )

#    (rbmClassif, y_hat_rbm, error) = restrictedBoltzmanMachine(X_raw, y, X_test)
    print( "RBM accuracy : " + str(error) )

    # ==== USE PREVIOUS ESTIMATES AS INPUT ====
    # IE : use classifiers as a dimension reduction
    X_estim = [lda.predict(X_afterPca),
               regLog.predict(X_afterPca),
               knn.predict(X_afterPca),
               forest.predict(X_afterPca)]
    X_estim = list(map(lambda *row: list(row), *X_estim)) # Transpose
    (forestEstim, y_hat_estim, error) = randomForest(X_estim, y, X_estim)
    print( "Classify from classifiers accuracy : " + str(error) )

#    (svm_estim, error) = logisticReg(X_estim, y)
#    print( "Classify from estimates : " + str(error) )
#    result = svm_estim.predict(X_test)
#    saveArrayToFile(result, "ClassifyFromEstimates")


    # Job done -> send email to see it on the smartphone !!! G33k :p
#    sendEmail()



#########################"

#sendEmail()
#exit()
#######################


if __name__ == "__main__":
    main()

