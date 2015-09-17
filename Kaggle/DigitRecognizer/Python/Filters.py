#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt

from sklearn.decomposition import PCA



def compute_Z_score(X, y):
    N_features = X.shape[1]

    # === Feature selection : compute Z-score
    XtX = np.dot( X.T, X )
    det = np.linalg.det( XtX )
    print("det( XtX ) = " + str(det))
    assert(det > 1e-3)
    # assert det not null, ie : matrix XtX not singular
    # assert that columns of X_after_PCA are not linearly dependants
    beta_hat = np.linalg.inv( XtX ).dot( X.T).dot(y)
    print( beta_hat )

    zScores = np.zeros( N_features )
    diago = np.diag( XtX )
    assert(len(diago) == N_features)
    sigma = np.std(X, axis=0)
    assert(len(diago) == len(sigma))
    for i in range( N_features ):
        zScores[i] = beta_hat[i] / (sigma[i] * sqrt(diago[i]))
    print("Z-scores : " + str(len(zScores)))
    print( zScores )

    # Add column number to keep columns with Z-Score > 2
    zScore = [ [i, zScores[i]] for i in range( N_features ) ]
    print( zScore[0:10] )

    zScore.sort(key=lambda x: abs(x[1]) ) # list of [column_index, Z-score]
    columnsToKeep = [ x[0] for x in zScore if abs(x[1]) > 2 ]
    print( columnsToKeep )
    print( "Z-Score => Number of columns to keep : " + str(len(columnsToKeep)))

    X_ZScore = X[:, columnsToKeep]

    return X_ZScore, zScores[columnsToKeep]


def pca100pourcent(X):
    """ Apply PCA on X and keep only columns with information
    ie : eigenvalue > 1e-5
    """
    pca = PCA(784)
    pca.fit( X )
    eigenVal = [ x for x in pca.explained_variance_ if abs(x) > 1e-5 ]
    print("We can keep {0} components without loosing information".format(len(eigenVal)) )

    # Keep same information in dataset, remove useless parameters (parameters with variance < 1e-5)
    N_features = len(eigenVal)
    pca = PCA( N_features )
    X_after_PCA = pca.fit_transform( X )
    # if keep 640 components : det( X_after_PCA ) = 2.17628211139e-114
    # if keep 635 components : det( X_after_PCA ) = 1.42486552624e+22
    # if keep 630 components : det( X_after_PCA ) = 3.59056547778e+157
    # if keep 590 components : det( X_after_PCA ) = np.inf, variance > 0.99
    print("mean(X) = {0}, std(X) = {1}".format(
        max(abs(np.mean(X_after_PCA, axis=0))),
        max(np.std( X_after_PCA, axis=0))
    ))

    return pca, X_after_PCA