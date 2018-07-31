import pandas as pd

################ For Regression ################











################ For Classification ################



'''
GBM stands for Grandient Boosting
    - deal with bias variance trade-off
    - boosting is a sequential technique

XGBoost stands for eXtreme Gradient Boosting
    - an advanced implementation of gradient boosting algorithm


parameters: 1) tree-specific params; 2) boosting params; 3) miscellaneous params;

Tree-specific params:
    - min_samples_split: 0.5-1% of total values
    - min_samples_leaf: used to control over-fitting
    - min_weight_fraction_leaf: similar to min_samples_leaf, just pick one
    - max_depth: 5 - 8
    - max_leaf_nodes: maximum number of terminal nodes or leaves, if this defined, max_depth will be ignore
    - max_features: the number of features to consider while searching for a best split;
        rule of thumb: square root of the total num of features 'sqrt'

Boosting params:
    - learning rate: lower values are preferred; 0.05 - 0.2
    - n_estimators: the number of sequential trees to be modeled; 40 - 70
    - subsample: the fraction of obs to be selected for each tree by random sampling
        typical value ~ 0.8

Miscellaneous params:
    - loss
    - init
    - random_state
    - verbose: 0 is no output, 1 is output generated for trees in certain intervals, >1 output generate for all trees
    - warm_start
    - presort

Tips:
    - fix learning rate and # of estimators for tuning tree-based params
    - tuning tree-specific params
        : tune max_depth and num_samples_split
        : tune min_samples_leaf
        : tune max_features
    - tuning subsample and making models with lower learning rate
'''


'''
References:
1. https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
2.


'''