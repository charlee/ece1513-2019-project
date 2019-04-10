import sys
import numpy as np
import argparse
from dataset import SVHN
from models import Model
from models.layers import Conv2D, BatchNorm, MaxPooling2D, Flatten, Dense, Dropout


def load_data():
    svhn = SVHN()
    X_train, y_train = svhn.load_preprocessed_data('train')
    X_test, y_test = svhn.load_preprocessed_data('test')

    # svhn.visualize(X_train, y_train)

    return X_train, y_train, X_test, y_test

def make_nn(model_path):
    nn = Model(model_path)
    nn.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Flatten(),
            Dense(32 * 32, activation='relu'),
            Dense(10),
        ],
        alpha=1e-4,
    )

    return nn


def make_cnn(model_path):
    cnn = Model(model_path)

    cnn.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Conv2D(32, (3, 3), activation='relu'),
            BatchNorm(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10),
        ],
        alpha=1e-3,
    )

    return cnn


def make_simplenet(model_path):
    cnn = Model(model_path)

    cnn.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #1
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #2
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #3
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #4
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #5
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #6
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #7
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #8
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #9
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', batch_normal=0.95),     #10
            Conv2D(256, (1, 1), activation='relu', batch_normal=0.95),     #11
            Conv2D(64, (1, 1), activation='relu', batch_normal=0.95),      #12
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #13
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10),
        ],
        alpha=1e-3,
    )

    return cnn

def make_simplenet_dropout(model_path):
    cnn = Model(model_path)

    cnn.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #1
            Dropout(0.8),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #2
            Dropout(0.8),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #3
            Dropout(0.8),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #4
            Dropout(0.8),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #5
            Dropout(0.8),
            Conv2D(32, (3, 3), activation='relu', batch_normal=0.95),      #6
            Dropout(0.8),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #7
            Dropout(0.8),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #8
            Dropout(0.8),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #9
            Dropout(0.8),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', batch_normal=0.95),     #10
            Dropout(0.8),
            Conv2D(256, (1, 1), activation='relu', batch_normal=0.95),     #11
            Dropout(0.8),
            Conv2D(64, (1, 1), activation='relu', batch_normal=0.95),      #12
            Dropout(0.8),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', batch_normal=0.95),      #13
            Dropout(0.8),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10),
        ],
        alpha=1e-3,
    )

    return cnn

def make_lr(model_path):
    lr = Model(model_path)
    lr.build_graph(
        image_shape=(32, 32, 1),
        n_classes=10,
        layers=[
            Flatten(),
            Dense(10),
        ],
        alpha=1e-3
    )

    return lr


if __name__ == '__main__':

     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs=1, choices=['lr', 'cnn', 'nn', 'simplenet', 'simplenet-dropout'],
                        required=True, help='Hidden layer size.')
    parser.add_argument('--path', type=str, nargs=1,
                        required=True, help='Model and output data save path.')
    parser.add_argument('--epochs', type=int, nargs=1,
                        default=[200], help='Training epoches.')
    parser.add_argument('--predict', action='store_true', help='Make prediction and plot confusion matrix.')
    args = parser.parse_args()

    if args.model[0] == 'lr':
        model = make_lr(args.path[0])
    elif args.model[0] == 'nn':
        model = make_nn(args.path[0])
    elif args.model[0] == 'cnn':
        model = make_cnn(args.path[0])
    elif args.model[0] == 'simplenet':
        model = make_simplenet(args.path[0])
    elif args.model[0] == 'simplenet-dropout':
        model = make_simplenet_dropout(args.path[0])
    else:
        print('Wrong model name!')
        sys.exit(1)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Reshape and normalize
    X_train = np.reshape(X_train, (*X_train.shape, 1)) / 255.0
    X_test = np.reshape(X_test, (*X_test.shape, 1)) / 255.0

    y_train[y_train == 10] = 0
    y_train = np.reshape(y_train, (-1,))

    y_test[y_test == 10] = 0
    y_test = np.reshape(y_test, (-1,))

    # Train
    model.set_data(X_train, y_train, X_test, y_test)
    model.init_session()

    if args.predict:
        from cm import plot_confusion_matrix
        import matplotlib.pyplot as plt
        labels = np.array([str(i) for i in range(10)])
        y_ = model.run_predict()
        plot_confusion_matrix(y_test, y_, labels, title='Confusion Matrix (model=%s)' % args.model[0])
        plt.savefig('%s/%s-cm.png' % (args.path[0], args.model[0]))
    else:
        model.train(batch_size=100, epochs=args.epochs[0])
