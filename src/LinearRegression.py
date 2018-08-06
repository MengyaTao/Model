
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# models
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# metrics
from sklearn.metrics import mean_squared_error, r2_score



# use sklearn
def sklearn_linearRegression(X, y):
    # lm = LinearRegression(fit_intercept = False)
    model = LinearRegression(fit_intercept=True) # initiate the model
    model.fit(X, y)
    y_pred = model.predict(X)
    # print predictions[:5]
    # score = model.score(X, y) # R^2 of the model
    # coef = model.coef_ # the coefficients for the predictions, an array
    # intercept = model.intercept_
    print_model_statistics(model, y, y_pred)
    # construct a data frame that contains features and estimated coefficients
    df = pd.DataFrame(zip(X.columns, model.coef_), columns = ['features', 'estimatedCoefficients'])
    print df
    return model


# polynomial basis function
def sklearn_polynomialRegression(X, y, degree=2, **kwargs):
    # convert one-dim array into a 3-dim array by taking the exponent of each value
    # poly = PolynomialFeatures(3, include_bias=False)
    model = make_pipeline(PolynomialFeatures(degree), # 7-degree polynomical model
                               LinearRegression(**kwargs))

    model.fit(X, y)
    y_pred = model.predict(X)
    print_model_statistics(model, y, y_pred)
    return model

class GuassianFeatures(BaseEstimator, TransformerMixin):
    # uniformly spaced Guassian features for one-dim input
    def __int__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x-y)/width
        return np.exp(-0.5 * np.sum(arg**2, axis))

    def fit(self, X, y=None):
        self.centers_ = np.linespace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1]-self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(
            X[:,:,np.newaxis], self.centers_, self.width_, axis=1
        )

# gaussian basis function
def sklearn_gaussianRegression(X, y):
    xfit = np.linspace(0, 10, 1000)
    model = make_pipeline(GuassianFeatures(20),
                                LinearRegression())
    model.fit(X[: np.newaxis], y)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.scatter(X,y)
    plt.plot(xfit, yfit)
    plt.xlim(0, 10)
    return model



# regression with regularization
## ridge regression (L2 or Tikhonov, penalizing the sum of squares of coefficients)
## generally works well even in presence of highly correlated features
def sklearn_ridgeRegression(X, y, alpha):
    model = Ridge(alpha)
    # build in CV: RidgeCV(alphas=[0.1, 1.0, 10])
    model.fit(X, y)
    predictions = model.predict(X)



## lasso regression estimates sparse coefficients, prefer fewer params
## penalizing the absolute value of coefficients
## arbitrarily selects any one feature among the highly correlated ones and reduce others to 0
def sklearn_lassoRegression(X, y, alpha):
    model = Lasso(alpha)
    # build in CV: LassoCV and LassoLarsCV
    # LassoLarsCV is faster than LassCV if # of obs is small compared to # of featurs
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions



# elastic net is trained with L1 and L2 prior as regularizer, allow learning a sparse model
# where few of the weights are non-zero like Lasso while still maintain regularization of Ridge

# use statsmodel
def statsmodel_fit_withoutConstant(X, y):
    # X and y are all in the df format
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    # print out the statistics
    # print model.summary()
    return predictions


def statsmodel_fit_withConstant(X, y):
    X = sm.add_constant(X) # add the intercept
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    return predictions


def print_model_statistics(model, y_true, y_pred):
    print 'Coefficients: \n', model.coef_
    print 'Mean squared error: %.2f' % mean_squared_error(y_true, y_pred)
    print 'Variance score: %.2f' % r2_score(y_true, y_pred)

'''
Statistics Interpretation
    : OLS - ordinary least squares
    : R^2 - the percentage of variance the model explains



'''