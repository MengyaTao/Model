import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import DataPreprocessing

# visualize distribution of each numerical feature
def plot_dist(df, var_list, output_path, fig_name):
    temp = pd.melt(df, value_vars=var_list)
    grid = sns.FacetGrid(temp, col='variable',
                         col_wrap=6, size=3.0, aspect=0.8,
                         sharex = False, sharey=False)
    grid.map(sns.distplot, "value")
    plt.savefig(output_path + fig_name + '.png')




input_file = '../data/input/house_prices/train.csv'
df, quan_list, qual_list = DataPreprocessing.quan_qual_feature_division(input_file, features_to_drop_list=['Id'])
output_path = '../data/figure/'
fig_name = 'dist_1'
plot_dist(df, quan_list, output_path, fig_name)