import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score # accuracy classification score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC # classifier


# split the data (holdout sets)
def sklearn_train_test_split(X, y, test_size_value, random_state_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=random_state_value)
    return X_train, X_test, y_train, y_test


def sklearn_cv_kfold(model, X, y, kfold = 5):
    score_array = cross_val_score(model, X, y, cv=kfold)
    return score_array



def sklearn_cv_leaveOneOut(model, X, y):
    score_array = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))
    print score_array.mean()
    return score_array


# validation curve
# train_score, val_score = validation_curve(PolynomicalRegression(), X, y, 'polynomicalfeatures_degree', degree, cv=7)

# grid search cv
def sklearn_cv_gridSearch(X, y, X_test, y_test):
    # assume this model is SVC
    model = SVC(kernel='rbf', class_weight='balanced')
    param_grid = {
        'svc_C': [1, 5, 10],
        'svc_gamma': [0.0001, 0.001, 0.01]
    }
    grid = GridSearchCV(model, param_grid)
    grid.fit(X, y)
    print grid.best_params_
    model_best = grid.best_estimator_
    yfit = model_best.predict(X_test)
    return model_best

'''
bias - variance trade-off
    bias: underfitting
    variance: overfitting
'''