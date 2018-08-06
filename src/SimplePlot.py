import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve

rng = np.random.RandomState(1)

x = 10*rng.rand(50)
y = 2*x-5+rng.randn(50)

x = np.array([2,3,4])
print x

def plot_scatter(X, y, fig_name):
    plt.scatter(X, y)
    # plt.plot(xfit, yfit) # plot the line
    plt.savefig(fig_name + '.png')



# residual plot
def plot_scatter_residuals(X, y, model):
    x_data = model.predict(X)
    y_data = model.predict(X) - y
    plt.scatter(x_data, y_data, c='b', s=40, alpha=0.5)
    # if test set also want to be printed out
    # plt.scatter(lm.predict(X_test), lm.predict(X_test)-y_test, c='g', s=40)
    plt.hlines(y = 0, xmin=0, xmax=X.max())
    plt.show()

# plot_scatter(x,y, 'test')

