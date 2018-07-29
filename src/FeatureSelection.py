import pandas as pd
import numpy as np


from xgboost import XGBRegressor


from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def generate_feature_importance(X_train, y_train):
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    imp = pd.DataFrame(xgb.feature_importances_, columns=['Importance'],
                       index = X_train.columns)
    imp = imp.sort_values(['Importance'], ascending=False)
    return imp



def calculate_rmse(y_true, y_pred):
    rmse_value = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse_value


def calculate_rmse_with_cv(model, n_folds, X_train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring='neg_mean_squared_error',
                                    cv = kf))
    return rmse

# define a funciton to calculate negative RMSE (as a score)
def calculate_neg_rmse(y_true, y_pred):
    neg_rmse_value = -1 * calculate_rmse(y_true, y_pred)
    return neg_rmse_value


def remove_redundant_features(X_train, y_train, X_test, neg_rmse):
    estimator = XGBRegressor()
    selector = RFECV(estimator, cv = 3,
                     n_jobs=-1, scoring=neg_rmse)
    selector = selector.fit(X_train, y_train)

    print 'The number of selected features is : {}'.format(selector.n_features_)
    features_kept = X_train.columns.values[selector.support_]
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    # transfrom it to a numpy array to feed into a neural network
    y_train = y_train.values
    return X_train, y_train, X_test, features_kept

