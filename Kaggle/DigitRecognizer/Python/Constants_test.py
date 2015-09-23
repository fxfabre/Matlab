#!/usr/bin/python3
# -*- coding: utf-8 -*-



#   Variance 0.7  => n components : 26
#   Variance 0.75 => n components : 33
#   Variance 0.8  => n components : 43
#   Variance 0.85 => n components : 59
#   Variance 0.9  => n components : 86
#   Variance 0.95 => n components : 152
#   Variance 0.97 => n components : 211
#   Variance 0.99 => n components : 327


PCA_VARIANCES       = [0.80]
RBM_N_COMPONENTS    = [20]
RBM_LEARNING_RATE   = [0.1]
ICA_N_COMPONENTS    = [20]
GRAD_NMF_N_COMPONENTS = [20]
REGLOG_C            = [1]
SVM_C               = [1]


LDA_PARAMS = [{
    'solver'        : ['svd'],
    'n_components'  : PCA_VARIANCES
},
{
    'solver'        : ['lsqr', 'eigen'],
    'n_components'  : PCA_VARIANCES,
    'shrinkage'     : [1]
}]


REGLOG_PARAM = [{
    'penalty'   : ['l1'],
    'C'         : REGLOG_C,
    'solver'    : ['liblinear']
}]


RBM_PARAM = [{
    'penalty'   : ['l1'],
    'C'         : REGLOG_C,
    'solver'    : ['liblinear']
},
{
    'penalty'   : ['l2'],
    'C'         : REGLOG_C,
    'solver'    : ['newton-cg']
}]


KNN_PARAM = {
    'n_neighbors'  : [15],
    'weights'      : ['uniform'],
    'algorithm'    : ['kd_tree'],
    'leaf_size'    : [100],
    'metric'       : ['minkowski'],
    'p'            : [1]
}


RANDOM_FOREST_PARAM = {
    'n_estimators'  : [5],
    'max_leaf_nodes': [100]
}


SVM_PARAM = [{
    'C'            : SVM_C,
    'kernel'       : ['linear']
},
{
    'C'            : SVM_C,
    'kernel'       : ['sigmoid']
}]






