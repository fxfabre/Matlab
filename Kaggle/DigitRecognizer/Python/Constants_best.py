#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

#   Variance 0.7  => n components : 26
#   Variance 0.75 => n components : 33
#   Variance 0.8  => n components : 43
#   Variance 0.85 => n components : 59
#   Variance 0.9  => n components : 86
#   Variance 0.95 => n components : 152
#   Variance 0.97 => n components : 211
#   Variance 0.99 => n components : 327


PCA_VARIANCES       = 0.80
RBM_N_COMPONENTS    = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
RBM_LEARNING_RATE   = [0.001, 0.01, 0.1, 1]
ICA_N_COMPONENTS    = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
GRAD_NMF_N_COMPONENTS = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
REGLOG_C            = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]
SVM_C               = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]


LDA_PARAMS = {
    'solver'    : 'lsqr',
    'shrinkage' : 'auto'
}


REGLOG_PARAM = [{
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


RBM_PARAM = [{
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


KNN_PARAM = {
    'n_neighbors'  : 15,
    'weights'      : 'uniform',
    'algorithm'    : 'kd_tree',
    'leaf_size'    : 30,
    'metric'       : 'minkowski',
    'p'            : 1
}


RANDOM_FOREST_PARAM = {
    'n_estimator'   : '100',
    'max_leaf_node' : '200'
}


SVM_PARAM = [{
    'C'            : SVM_C,
    'kernel'       : ['linear']
},
{
    'C'            : SVM_C,
    'kernel'       : ['poly'],
    'degree'       : [1, 2, 3, 4],
    'gamma'        : [0.001, 0.01, 0.1, 0.5, 1]
},
{
    'C'            : SVM_C,
    'kernel'       : ['rbf', 'sigmoid']
}]





