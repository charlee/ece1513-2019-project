import numpy as np
from dataset import SVHN
from models.cnn import CNN
from models.layers import Conv2D, BatchNorm, MaxPooling2D, Flatten, Dense, Dropout


def load_data():
    svhn = SVHN()
    X_train, y_train = svhn.load_preprocessed_data('train')
    X_test, y_test = svhn.load_preprocessed_data('test')

    # svhn.visualize(X_train, y_train)

    return X_train, y_train, X_test, y_test


def make_cnn(model_path):
    cnn = CNN(model_path)

    cnn.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Conv2D(32, (3, 3), activation='relu'),
            BatchNorm(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(32 * 32, activation='relu'),
            Dropout(0.5),
            Dense(10),
        ],
        alpha=1e-6,
    )

    X_train, y_train, X_test, y_test = load_data()

    X_train = np.reshape(X_train, (*X_train.shape, 1))
    X_test = np.reshape(X_test, (*X_test.shape, 1))

    y_train[y_train == 10] = 0
    y_train = np.reshape(y_train, (-1,))

    y_test[y_test == 10] = 0
    y_test = np.reshape(y_test, (-1,))

    cnn.set_data(X_train[:1000], y_train[:1000], X_test[:100], y_test[:100])
    cnn.init_session()
    cnn.train()


if __name__ == '__main__':
    make_cnn('./__model__')
    # X_train, y_train, X_test, y_test = load_data()

