import sys
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

def run(model, df_train, n_folds=5):
    print 'reading csv...'
    max_timesteps = 10

    df_train = pd.read_csv('train.csv', nrows=n_rows)
    df_train = df_train[df_train['is_booking'] == 1]

    print 'preparing data..'

    #X, y = prep_data_for_rnn(df_train, max_timesteps)
    X, y, weights = prep_data(df_train)
    kf = KFold(X.shape[0], n_folds=n_folds)

    for fold_i, (train, test) in enumerate(kf):
        print "Fold {}".format(fold_i + 1)

        x_train, y_train = X[train], y[train]
        x_test, y_test = X[test], y[test]

        X_train = []
        for x in x_train:
            X_train.append([x])

        X_test = []
        for x in x_test:
            X_test.append([x])

        model.train()
