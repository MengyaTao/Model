import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

def build_model_5_ensembles(X_train):
    # build the model with already tuned hyperparameters
    # XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, subsample=0.5,
                       colsample_bytree=-.5, max_depth=3, gamma=0, reg_alpha=0,
                       reg_lambda=2, min_child_weight=1)

    # Lasso
    las = Lasso(alpha=0.00049, max_iter=50000)

    # Elastic Net
    elast = ElasticNet(alpha=0.0003, max_iter=50000, l1_ratio=0.83)

    # Kernel Ridege
    ridge = KernelRidge(alpha=0.15, coef0=3.7, degree=2, kernel='polynomial')

    # Gradient Boosting
    boost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.042, subsample=0.5,
                                      random_state=0, min_samples_split=4, max_depth=4)

    # Neural Network
    nn = Sequential()
    # Layers
    nn.add(Dense(units=40, kernel_initializer='uniform', activation='relu',
                 input_dim=X_train.shape[1], kernal_regularizer = regularizers.l2(0.003)))
    nn.add(Dense(units=20, kernel_initializer='uniform', activation='relu',
                 kernal_regularizer=regularizers.l2(0.003)))
    nn.add(Dense(units=20, kernel_initializer='uniform', activation='relu',
                 kernal_regularizer=regularizers.l2(0.003)))
    nn.add(Dense(units=1, kernel_initializer='uniform', activation='relu',
                 kernal_regularizer=regularizers.l2(0.003)))
    # compile the NN
    nn.compile(loss='mean_squared_error', optimizer='sgd')

    model_list = [xgb, las, ridge, boost, nn]
    return model_list


class Ensemble(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def level0_to_level1(self, X):
        self.predictions_ = []
        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X).reshape(X.shape[0], 1))
        return np.concatenate(self.predictions_, axis=1)

    def fit(self, X, y):
        for regressor in self.regressors:
            if regressor != nn:
                regressor.fit(X, y)
            else:
                regressor.fit(X, y, batch_size=64, epochs=1000, verbose=0)  # Neural Network
        self.new_features = self.level0_to_level1(X)

        # using a large L2 regularization to prevent the ensemble from biasing toward
        # one particular base model
        self.combine = Ridge(alpha=10, max_iter=50000)
        self.combine.fit(self.new_features, y)
        self.coef_ = self.combine.coef_

    def predict(self, X):
        self.new_features = self.level0_to_level1(X)
        return self.combine.predict(self.new_features).reshape(X.shape[0])



def use_ensemble(X_train, y_train, X_test):
    model_list = build_model_5_ensembles(X_train)
    model = Ensemble(regressors = model_list)
    model.fit(X_train, y_train)
    y_pred = np.exp(model.predict(X_test))
    print("\nThe weights of those base models are: {}".format(model.coef_))

    output = pd.DataFrame({'predicted y value': y_pred})
    output.to_csv('prediction-ensemble.csv', index=False)
    return y_pred


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

    # use of this class
    # averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
    # score = rmsle_cv(averaged_models)