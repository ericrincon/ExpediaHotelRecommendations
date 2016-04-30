from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers.recurrent import LSTM, GRU

from keras.optimizers import SGD

import sys
import pandas as pd
import numpy as np
import getopt
import sys

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

def prep_data(data_frame, des_path):
    destinations_df = pd.read_csv(des_path + 'destinations.csv')
    srch_destination_ids = destinations_df['srch_destination_id']
    search_d_ids = data_frame['srch_destination_id']

    n_ids = search_d_ids.max() + 1
    print 'N IDs: {}'.format(n_ids)
    weights = np.zeros((n_ids, 150))

    for i in range(srch_destination_ids.shape[0]):
        id = srch_destination_ids[i]

        weight = destinations_df[destinations_df['srch_destination_id'] == id]

        weights[i, :] = weight

    labels = data_frame['hotel_cluster'].as_matrix()

    X = search_d_ids.as_matrix()
    y = np.zeros((X.shape[0], 100))

    for i in range(X.shape[0]):
        label = labels[i]
        y[i, label] = 1

    return X, y, weights


def build_nn_model(vector_len, weights, dense_layers, activation='relu', dropout_rate=.5, init_dropout_rate=.2):
    model = Sequential()
    model.add(Embedding(weights.shape[0], vector_len, input_length=1, weights=[weights]))
    model.add(Flatten())
    model.add(Dropout(init_dropout_rate))

    for layer_size in dense_layers:
        model.add(Dense(layer_size))
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(100))
    model.add(Activation('softmax'))

    return model


def main():
    n_folds = 5

    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['activation=', 'dropout_rate=', 'dense_layers=', 'n_rows=',
                                                      'destinations_path=', 'train_path=', 'test_path='])
    except getopt.GetoptError as error:
        print error
        sys.exit(2)

    activation = 'relu'
    dropout_rate = .5
    dense_layers = [200, 200]
    n_rows = 100000
    train_path = ''
    test_path = ''
    destinations_path = ''

    for opt, arg in opts:
        if opt == '--activation':
            activation = arg
        elif opt == '--dropout_rate':
            dropout_rate = int(arg)
        elif opt == '--model_type':
            model_type = arg
        elif opt == '--dense_layers':
            dense_layers = arg.split(',')

        elif opt == '--destinations_path':
            destinations_path = arg
        elif opt == '--train_path':
            train_path = arg
        elif opt == '--test_path':
            test_path

        else:
            print "Option {} is not valid!".format(opt)


    print 'reading csv...'
    max_timesteps = 10

    df_train = pd.read_csv(train_path + 'train.csv', nrows=20000000)
    df_train = df_train[df_train['is_booking'] == 1]

    print 'preparing data..'

    #X, y = prep_data_for_rnn(df_train, max_timesteps)
    X, y, weights = prep_data(df_train, destinations_path)
    kf = KFold(X.shape[0], n_folds=5)

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

        model = build_nn_model(150, weights, dense_layers, activation=activation, dropout_rate=dropout_rate)
        model.compile('adam', 'categorical_crossentropy')

        model.fit(X_train, y_train)

        predictions = model.predict_classes(X_test)
        y_test = np.argmax(y_test, 1)
        accuracy = accuracy_score(y_test, predictions)

        print "Accuracy: {}\n".format(accuracy)
        sys.exit()

# 12 - 13


if __name__ == '__main__':
    main()