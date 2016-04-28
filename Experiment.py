from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU

from keras.optimizers import SGD

import sys
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier


def build__nn_model():
    model = Sequential()
    model.add(Dense(100, input_dim=18))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(100))
    model.add(Activation('softmax'))

    return model

def build_rnn_model(n_features):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(1, )))
    model.add(Dropout(.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(100))
    model.add(Activation('softmax'))

    return model

def main():
    print 'reading csv...'
    df_train = pd.read_csv('train.csv', nrows=1000)

    X, y = prep_data(df_train)

    kf = KFold(X.shape[0], n_folds=5)


    for fold_i, (train, test) in enumerate(kf):
        print "Fold {}".format(fold_i + 1)

        X_train, y_train = X[train, :], y[train]
        X_test, y_test = X[test, :], y[test]

        """
        model = build_model()

        sgd = SGD(lr=.001)

        model.compile('adam', 'categorical_crossentropy')
        model.fit(X_train, y_train, nb_epoch=30, validation_split=.2)
        predictions = model.predict_classes(X_test)
        """

        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        #y_test = np.argmax(y_test, 1)
        accuracy = accuracy_score(y_test, predictions)

        print "Accuracy: {}\n".format(accuracy)
        sys.exit()

# 12 - 13
def prep_data(data_frame):

    destinations_df = pd.read_csv('destinations.csv')

    # Fix the date and time
    date_time = data_frame['date_time']
    dates = []
    times = []

    for i in range(date_time.shape[0]):
        dt = date_time.iloc[i]
        date, time = dt.split()

        dates.append(date)
        times.append(time)

    # Add to the dataframe
    dates = pd.Series(dates)
    times = pd.Series(times)

    data_frame['time'] = times
    data_frame['date'] = dates

    # Remove the old date and time column from the dataset
    data_frame = data_frame.drop('date_time', 1)

    # Sort by user id
    user_ids = data_frame['user_id'].unique()

    sessions = []
    session_labels = []

    for user_id in user_ids:
        user_sessions = data_frame[data_frame['user_id'] == user_id]
        user_dates = user_sessions['date'].unique()

        for user_date in user_dates:
            user_date_sessions = user_sessions[user_sessions['date'] == user_date]


            user_date_sessions = user_date_sessions.sort_values('time')

            #user_date_sessions = user_date_sessions.drop('time', 1)
            #user_date_sessions = user_date_sessions.drop('date', 1)

            labels = user_date_sessions['hotel_cluster'].as_matrix()
            session_labels.append(labels)
            sessions.append(user_date_sessions)

    for i in range(10):
        print '{}'.format(sessions[i])
        print 'labels: {}'.format(session_labels[i])
        print '\n\n'
    sys.exit()
    data_frame['orig_destination_distance'] = 0

    X = data_frame.iloc[:, range(1, 10) + range(13, 22)].as_matrix()




    """
    y = np.zeros((X.shape[0], 100))

    for i in range(X.shape[0]):
        label = labels[i]
        y[i, label] = 1
    """
    y = labels

    return X, y

if __name__ == '__main__':
    main()