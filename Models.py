from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers.recurrent import LSTM, GRU

class RNN():
    def build_model(self, n_features, n_timesteps):
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(.5))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dropout(.5))
        model.add(Dense(100))
        model.add(Activation('softmax'))

        return model

class NN():
    def build_nn_model(self, vector_len, weights, dense_layers, activation='relu', dropout_rate=.5,
                       init_dropout_rate=.2):
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
