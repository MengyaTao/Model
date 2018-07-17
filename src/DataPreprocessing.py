import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew


def quan_qual_feature_division(input_file, features_to_drop_list):
    # divide the features to numerical and categorical
    # the output are the list of column names belongs to each type
    df = pd.read_csv(input_file)
    print df.shape
    quan_list = list(df.loc[:, df.dtypes != 'object'].drop(features_to_drop_list, axis=1).columns.values)
    qual_list = list(df.loc[:, df.dtypes == 'object'].columns.values)
    print 'Quantitative features are: ', quan_list, ' ', len(quan_list)
    print 'Qualitative features are: ', qual_list, ' ', len(qual_list)

    return df, quan_list, qual_list


def get_nan_cols(df, quan_list, qual_list):
    hasNAN = df[quan_list].isnull().sum()
    hasNAN = hasNAN[hasNAN > 0]
    hasNAN = hasNAN.sort_values(ascending=False)
    print(hasNAN)

    print('**' * 40)

    hasNAN = df[qual_list].isnull().sum()
    hasNAN = hasNAN[hasNAN > 0]
    hasNAN = hasNAN.sort_values(ascending=False)
    print(hasNAN)


def fill_in_nan(df, col_list):
    for col in col_list:
        # fill in the data with median values - numerical feature
        df[col] = df[col].fillna(df[col].median(), inplace=True)
        # fill in the data with zero  - numerical feature
        # df[col] = df[col].fillna(0, inplace=True)

        # fill in the data with 'NA' or 'None' - categorical feature
        # df[col] = df[col].fillna('None', inplace=True)
        # fill in the data with mode value - categorical feature
        # df[col] = df[col].fillna(df[col].mod()[0], inplace=True)

    return df


def transform_qual_features_to_quan(df):
    df = df.map({'NA':0, 'Grvl':1, 'Pave':2})

    return df


def visualize_quan_list_hist(df, quan_list):
    # visualize the distribution of each numerical feature
    temp = pd.melt(df, value_vars=quan_list)
    grid = sns.FacetGrid(temp, col="variable", col_wrap=6, size=3.0,
                         aspect=0.8, sharex=False, sharey=False)
    grid.map(sns.distplot, "value")
    plt.show()


def visualize_quan_list_scatter(df, quan_list, y_var, y_label):
    # scatter plots
    temp = pd.melt(df, id_vars=[y_var], value_vars=quan_list)
    grid = sns.FacetGrid(temp, col="variable", col_wrap=4, size=3.0,
                         aspect=1.2, sharex=False, sharey=False)
    grid.map(plt.scatter, "value", y_label, s=1.5)
    plt.show()


def get_skewness_and_transform(df, quan_list):
    # print the skewness of each numerical feature
    for i in quan_list:
        print(i + ':', round(skew(df[i]), 2))

    # transform those with skewness > 0.5
    skewed_features = np.array(quan_list)[np.abs(skew(df[quan_list])) > 0.5]
    print skewed_features
    df[skewed_features] = np.log1p(df[skewed_features]) # log(1 + x), natural logarithm

    return df


def transform_dummy_variables(df, qual_list):
    # create of list of dummy variables that need to drop, which will be the last
    # column generated from each categorical feature
    dummy_drop = []
    for i in qual_list:
        dummy_drop += [i + '_' + str(df[i].unique()[-1])]

    # create dummy variables
    df = pd.get_dummies(df, columns=qual_list)
    # drop the last column generated from each categorical feature
    df = df.drop(dummy_drop, axis=1)

    return df

input_file = '../data/input/house_prices/train.csv'
df, quan_list, qual_list = quan_qual_feature_division(input_file, features_to_drop_list=['Id'])
# get_nan_cols(df, quan_list, qual_list)
# df = get_skewness_and_transform(df, quan_list)
df = pd.get_dummies(df, columns=qual_list)
print df.columns.values

# df.Age = df.Age.map(lambda x: 0 if x < 0 else x)