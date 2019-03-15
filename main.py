import numpy as np
from dataset import SVHN
from models.cnn import CNN
from models.layers import Conv2D, BatchNorm, MaxPooling2D, Flatten, Dense, Dropout


def load_data():
    svhn = SVHN()
    X_train, y_train = svhn.load_data('train')
    X_test, y_test = svhn.load_data('test')

    return X_train, y_train, X_test, y_test


def make_cnn(model_path):
    cnn = CNN(model_path)

    cnn.build_graph(
        image_shape=(32, 32, 3),
        n_classes=10,
        layers=[
            Conv2D(32, (3, 3), activation='relu'),
            BatchNorm(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(32 * 32, activation='relu'),
            Dropout(0.5),
            Dense(10),
        ]
    )

    X_train, y_train, X_test, y_test = load_data()
    y_train = np.reshape(y_train, (-1,))
    y_test = np.reshape(y_test, (-1,))
    cnn.set_data(X_train, y_train, X_test, y_test)
    cnn.init_session()
    cnn.train()




if __name__ == '__main__':
    make_cnn('./model')