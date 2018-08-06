import pandas as pd
import numpy as np
from numpy import nan

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer


# categorical data

# one-hot encoding, create extra columns indicating the presence and absence of a category
# this will increase the size of your datasets
def sklearn_dictVectorizer(df):
    vec = DictVectorizer(sparse=True, dtype=int)
    df = vec.fit_transform(df)
    # print vec.get_feature_names()
    return df


# sklearn.preprocessing.OneHotEncoder
# sklearn.feature_extraction.FeatureHasher




# imputation of missing data
# Pandas handles NaN (np.nan) and None the same way
def sklearn_impute_withMean(df):
    imp = Imputer(strategy='mean')
    df = imp.fit_transform(df)
    return df


# isnull(): generate a boolean mask indicating missing values
# notnull(): opposite of isnull()
# dropna(): return a filtered version of data
# fillna(): df.fillna(0), df.fillna(method='ffill') # forward-fill to propagate the previous value forward, or 'bfill'

# data[data.notnull()] - return the data not null = same as data.dropna()
# df.dropna() # by default, drop all rows in which any null value is present; df.dropna(axis='columns', how='all')
# df.dropna(axis='row', thresh=3) # specify a minimum # of non-null values for the row/column