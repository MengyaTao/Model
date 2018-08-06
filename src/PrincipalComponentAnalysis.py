import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set

from sklearn.decomposition import PCA, RandomizedPCA

def sklearn_PCA_fit(X, num_of_components):
    pca = PCA(n_components=num_of_components)
    pca.fit(X)
    print pca.components_
    print pca.explained_variance_
    X_pca = pca.transform(X)
    print 'original shape: ', X.shape
    print 'transformed shape: ', X_pca.shape
    # inverse transform: X_new = pca.inverse_transform(X_pca)
    return X_pca


def sklearn_PCA_get(X, explained_variance=0.5):
    pca = PCA(explained_variance).fit(X)
    print pca.n_components_



# RandomizedPCA is used for high-dim data
def sklearn_randomPCA(X, num_of_components):
    pca = RandomizedPCA(num_of_components)
    pca.fit(X)
    print pca.components_
    print pca.explained_variance_
    components = pca.transform(X)
    projected = pca.inverse_transform(components)


def plot_PCA(X, y):
    plt.scatter(X[:, 0], X[:, 1],
                c = y, edgecolors='none',
                alpha=0.5, cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


def plot_PCA_number_choose(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_))
    # or plt.plot(np.cumsum(pca.explained_variance_ratio))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()










'''
PCA - dimensionality reduction algorithm
    : can be useful as a tool for visualization, noise filtering, feature extraction/engineering

'''