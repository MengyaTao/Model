from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cross_validation import train_test_split


def split_df(X_df, y_df, test_size_value, random_state_value):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=test_size_value,
                                                        random_state=random_state_value)
    print ('# training samples: ', len(X_train))
    print ('# test samples: ', len(X_test))
    return X_train, X_test, y_train, y_test


def standardize_Xtrain_Xtest(X_train, X_test):
    # quan_list is the list of quantitative variables, df_Xtrain[quan_list]
    # if y is in the df, drop it
    # X_train = df[:1460].drop(['var1', 'var2'], axis=1)
    # fit the training set only and then transform both training and test sets
    scaler = RobustScaler()
    # or scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
