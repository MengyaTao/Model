import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# classification
def sklearn_decisionTree_classification(X, y):
    model = DecisionTreeClassifier().fit(X, y)
    return model


def sklearn_randomForest_classification(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    visualize_classifier(model, X, y)
    model.fit(X, y)
    y_pred = model.predict(X)
    print metrics.classification_report(y_pred, y)
    mat = confusion_matrix(y, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    return model

def visualize_classifier(model, X, y, sample_size, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=sample_size, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # create a color plot with the results
    nclasses = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels = np.arrange(nclasses+1)-0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
    ax.show()
    # visualize_classifier(DecisionTreeClassifier(), X, y)


# regression


'''
Random Forests - non-parametric algorithm
    : ensemble method
    : disadvantage - the results are not easily interpretable
'''