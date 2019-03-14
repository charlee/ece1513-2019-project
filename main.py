import numpy as np
from dataset import SVHN
from models.cnn import CNN, ConvLayer, ReLULayer, BatchNormLayer, PoolLayer, FlattenLayer, FCLayer, DropoutLayer


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
            ConvLayer(3, 32, 1),
            ReLULayer(),
            BatchNormLayer(),
            PoolLayer(2),
            FlattenLayer(),
            FCLayer(32 * 32),
            DropoutLayer(0.5),
            ReLULayer(),
            FCLayer(10),
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