from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import pandas
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score


def build_model():
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

def main():
    print 'reading csv...'
    df_train = pandas.read_csv('train.csv', nrows=100000)

    X, y = prep_data(df_train)

    kf = KFold(X.shape[0], n_folds=5)


    for fold_i, (train, test) in enumerate(kf):
        print "Fold {}".format(fold_i + 1)

        X_train, y_train = X[train, :], y[train]
        X_test, y_test = X[test, :], y[test]

        model = build_model()
        sgd = SGD(lr=.001)

        model.compile('adam', 'categorical_crossentropy')
        model.fit(X_train, y_train, nb_epoch=30, validation_split=.2)
        predictions = model.predict_classes(X_test)
        y_test = np.argmax(y_test, 1)
        accuracy = accuracy_score(y_test, predictions)

        print "Accuracy: {}\n".format(accuracy)

# 12 - 13
def prep_data(data_frame):
    X = data_frame.iloc[:, range(1, 10) + range(13, 22)].as_matrix()
    labels = data_frame.iloc[:, 23].as_matrix()
    y = np.zeros((X.shape[0], 100))

    for i in range(X.shape[0]):
        label = labels[i]
        y[i, label] = 1

    return X, y

if __name__ == '__main__':
    main()