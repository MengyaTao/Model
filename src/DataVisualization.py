import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import DataPreprocessing

# visualize distribution of each numerical feature
def plot_dist(df, var_list, output_path, fig_name):
    # var_list is the list of continuous variables - quantitative
    # change the format of the dataset to be 2 colns, one for 'variable', one for 'value'
    temp = pd.melt(df, value_vars=var_list)
    # col_wrap makes each row has 6 figures
    grid = sns.FacetGrid(temp, col='variable',
                         col_wrap=6, size=3.0, aspect=0.8,
                         sharex = False, sharey=False)
    grid.map(sns.distplot, "value")
    # plot the figure on screen
    # plt.show()
    # save the figure
    plt.savefig(output_path + fig_name + '.png')


# visualize distribution of each numerical feature against y value
def plot_scatter (df, y_var, var_list, y_label, output_path, fig_name):
    temp = pd.melt(df, id_vars=[y_var], value_vars=var_list)
    grid = sns.FacetGrid(temp, col='variable',
                         col_wrap=4, size=3.0, aspect=1.2,
                         sharex=False, sharey=False)
    grid.map(sns.scatter, "value", y_label, s=1.5)
    # plot the figure on screen
    # plt.show()
    # save the figure
    plt.savefig(output_path + fig_name + '.png')


def plot_missing_data_hist(df_na_top, output_path, fig_name):
    f, ax = plt.subplot(figsize=(15,12))
    plt.xticks(rotation='90')
    sns.barplot(x=df_na_top.index, y=df_na_top)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Percent of missing data', fontsize=12)
    plt.title('Percent of missing data by feature', fontsize=12)
    plt.show()
    # save the figure
    plt.savefig(output_path + fig_name + '.png')


def plot_correlation(df, output_path, fig_name):
    corrmat = df.corr()
    plt.subplot(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    # save the figure
    plt.savefig(output_path + fig_name + '.png')

input_file = '../data/input/house_prices/train.csv'
df, quan_list, qual_list = DataPreprocessing.quan_qual_feature_division(input_file, features_to_drop_list=['Id'])
output_path = '../data/figure/'
fig_name = 'dist_1'
plot_dist(df, quan_list, output_path, fig_name)