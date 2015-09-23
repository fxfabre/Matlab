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


PCA_VARIANCES       = [0.80, 0.85, 0.90, 0.93]
RBM_N_COMPONENTS    = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
RBM_LEARNING_RATE   = [0.001, 0.01, 0.1, 1]
ICA_N_COMPONENTS    = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
GRAD_NMF_N_COMPONENTS = [20, 25, 30, 35, 40, 50, 60, 70, 100, 150]
REGLOG_C            = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]
SVM_C               = [0.1, 0.25, 0.5, 0.75, 1.0, 3, 10, 100]


LDA_PARAMS = [{
    'solver'        : ['svd'],
    'n_components'  : PCA_VARIANCES
},
{
    'solver'        : ['lsqr', 'eigen'],
    'n_components'  : PCA_VARIANCES,
    'shrinkage'     : list(np.arange(0.125, 1.1, 0.125)) + [None, 'auto']
}]


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
    'n_neighbors'  : [7, 11, 15, 19],
    'weights'      : ['uniform', 'distance'],
    'algorithm'    : ['kd_tree', 'ball_tree'],
    'leaf_size'    : [15, 30, 50, 100],
    'metric'       : ['minkowski'],
    'p'            : [1, 2, np.inf]
}


RANDOM_FOREST_PARAM = {
    'n_estimators'  : [5,10,20,50,75,100],
    'max_leaf_nodes': [100, 200, 300, 400, 500]
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





