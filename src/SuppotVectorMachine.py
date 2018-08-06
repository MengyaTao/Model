import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.svm import SVC # classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# C is a hyperparameter, "softens" the margins
# very large C, the margin is hard and points cannot lie in it
# smaller C, the margin is softer and can grow to encompass some points
def sklearn_svc_linear(X, y):
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    print model.support_vectors_ # pivotal points
    return model



def sklearn_svc_kernal(X, y, ylabels):
    # rbf: radial basis function
    model = SVC(kernel='rbf', C=1E6)
    model.fit(X, y)
    y_pred = model.predict(X)
    # classification report: precision, recall, f1-score, support
    df = classification_report(y, y_pred, target_names=ylabels)
    print df

    # confusion matrix
    mat = confusion_matrix(y, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=ylabels, yticklabels=ylabels)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    return model


'''
SVM - maximize the margin
plt.fill_between(xfit, yfit-d, yfit+d, edgecolor='none', color='grey', alpha=0.4)
'''