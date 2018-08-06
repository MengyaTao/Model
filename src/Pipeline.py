from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, PolynomialFeatures


# example 1
def sklearn_pipeline_LR1(X, y):
    model = make_pipeline(Imputer(strategy='mean'),
                          PolynomialFeatures(degree=2),
                          LinearRegression())
    model.fit(X, y)
    y_pred = model.predict
    return model