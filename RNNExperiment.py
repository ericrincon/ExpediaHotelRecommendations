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



def main():
    rnn = build_rnn_model(171, max_timesteps)
    rnn.compile('adam', 'categorical_crossentropy')
    rnn.fit(X_train, y_train, nb_epoch=1)
    predictions = rnn.predict_classes(X_test)


    model = build_model()

    sgd = SGD(lr=.001)

    model.compile('adam', 'categorical_crossentropy')
    model.fit(X_train, y_train, nb_epoch=30, validation_split=.2)
    predictions = model.predict_classes(X_test)


    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

def prep_data_for_rnn(data_frame, max_timesteps):
    len_latent = 150
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

    average_session_length = 0

    for user_id in user_ids:
        user_sessions = data_frame[data_frame['user_id'] == user_id]
        user_dates = user_sessions['date'].unique()

        for user_date in user_dates:
            user_date_sessions = user_sessions[user_sessions['date'] == user_date]
            user_date_sessions = user_date_sessions.sort_values('time')
            average_session_length += user_date_sessions.shape[0]

            user_date_sessions = user_date_sessions.drop('time', 1)
            user_date_sessions = user_date_sessions.drop('date', 1)
            user_date_sessions = user_date_sessions.drop('srch_ci', 1)
            user_date_sessions = user_date_sessions.drop('srch_co', 1)

            label = user_date_sessions['hotel_cluster'].as_matrix()[-1]

            if user_date_sessions.shape[0] > max_timesteps:
                start = user_date_sessions.shape[0] - max_timesteps
                sliced_user_date_sessions = user_date_sessions.iloc[start:]
            else:
                sliced_user_date_sessions = user_date_sessions
            sessions.append(sliced_user_date_sessions)
            session_labels.append(label)

    vector_length = sessions[0].shape[1] + len_latent
    sessions_tensor = np.zeros((len(sessions), max_timesteps, vector_length))

    for i, user_sessions in enumerate(sessions):
        df_d149_matrix = np.zeros((user_sessions.shape[0], len_latent))

        for session_i in range(user_sessions.shape[0]):
            des_id = user_sessions.iloc[session_i]['srch_destination_id']
            df_d149 = destinations_df[destinations_df['srch_destination_id'] == des_id]

            if not df_d149.shape[0] == 0:
                df_d149_matrix[session_i, :] = df_d149.as_matrix()
        user_sessions = user_sessions.as_matrix()

        user_sessions = np.hstack((user_sessions, df_d149_matrix))

        if user_sessions.shape[0] < max_timesteps:
            padding_size = max_timesteps - user_sessions.shape[0]
            padding = np.zeros((padding_size, vector_length))

            user_sessions = np.vstack((padding, user_sessions))
        sessions_tensor[i, :, :] = user_sessions
    average_session_length /= len(sessions)

    X = sessions_tensor
    y = np.zeros((X.shape[0], 100))

    for i in range(X.shape[0]):
        label = session_labels[i]
        y[i, label] = 1

    return X, y

if __name__ == '__main__':
    main()