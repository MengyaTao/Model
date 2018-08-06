import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set

from sklearn.naive_bayes import GaussianNB, MultinomialNB

def sklearn_NB_classification(X, y):
    model = GaussianNB().fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return model





'''
Naive Bayes Classification
 : suitable for very high-dim datasets
 : fast, few tunable prams
 : quick and dirty baseline for a classification problem
 : make stringent assumptions about data, not perform as well as more complicated models

'''