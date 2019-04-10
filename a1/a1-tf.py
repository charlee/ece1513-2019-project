import time
import os
import sys
import argparse
import itertools
import numpy as np
import tensorflow as tf

from common import (
    loadData, PerfLogger, predict, accuracy, save_loss,
    save_model, preprocess_data, make_filenames,
    TimeThis, create_dir
)


def buildGraph(d, optimizer_type, loss_type, params):

    # Your implementation here
    X = tf.placeholder(tf.float32, shape=(None, d), name='X')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')
    w = tf.Variable(tf.truncated_normal((d,), stddev=0.5), name='w')
    # w = tf.Variable(tf.zeros(d,), tf.float32, name='w')
    reg = tf.placeholder(tf.float32, shape=(), name='reg')

    f = tf.matmul(X, tf.reshape(w, shape=(-1, 1)))
    y_ = tf.reshape(f, shape=(-1,))

    if loss_type == 'mse':
        # loss = tf.losses.mean_squared_error(y, y_)
        loss = tf.reduce_mean(tf.square(y - y_))
    elif loss_type == 'ce':
        loss = tf.losses.sigmoid_cross_entropy(y, y_)
    loss = loss + reg * tf.nn.l2_loss(w[1:])

    if optimizer == 'gd':
        opt_op = tf.train.GradientDescentOptimizer(
            learning_rate=params['alpha'],
        ).minimize(loss)
    else:
        opt_op = tf.train.AdamOptimizer(
            learning_rate=params['alpha'],
            beta1=params['beta1'],
            beta2=params['beta2'],
            epsilon=params['epsilon'],
        ).minimize(loss)

    predict = (tf.sign(y_ - 0.5) + 1) / 2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predict), tf.float32))

    return X, y, y_, w, reg, loss, predict, accuracy, opt_op


def SGD(X, y, Xv, yv, Xt, yt, epochs, batch_size, optimizer_type, loss_type, params):

    N, d = X.shape
    batch_count = int(np.ceil(X.shape[0] / batch_size))

    _X, _y, _y_, _w, _reg, _loss, _predict, _accuracy, _opt = buildGraph(
        d, optimizer_type, loss_type, params)

    perf_logger = PerfLogger([
        'train_loss', 'train_accuracy',
        'valid_loss', 'valid_accuracy',
        'test_loss', 'test_accuracy'
    ])

    tf.set_random_seed(421)
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(1, epochs+1):
            idx = np.arange(N)
            np.random.shuffle(idx)

            X = X[idx]
            y = y[idx]

            for j in range(batch_count):
                X_input = X[batch_size * j:batch_size * (j+1)]
                y_input = y[batch_size * j:batch_size * (j+1)]

                sess.run(_opt, feed_dict={_X: X_input,
                                          _y: y_input, _reg: params['reg']})

            if i % 10 == 0 or i == epochs:
                train_loss = sess.run(
                    _loss, feed_dict={_X: X_input, _y: y_input, _reg: params['reg']})
                train_accuracy = sess.run(_accuracy, feed_dict={
                                          _X: X_input, _y: y_input, _reg: params['reg']})
                valid_loss = sess.run(
                    _loss, feed_dict={_X: Xv, _y: yv, _reg: params['reg']})
                valid_accuracy = sess.run(_accuracy, feed_dict={
                                          _X: Xv, _y: yv, _reg: params['reg']})
                test_loss = sess.run(
                    _loss, feed_dict={_X: Xt, _y: yt, _reg: params['reg']})
                test_accuracy = sess.run(_accuracy, feed_dict={
                                         _X: Xt, _y: yt, _reg: params['reg']})
                perf_logger.append(i, {
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'valid_loss': valid_loss,
                    'valid_accuracy': valid_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                }, print_log=True)

    w = sess.run(_w)
    return w, perf_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, nargs=1, required=True,
                        choices=['gd', 'adam'], help='Sepcify the optimizer.')
    parser.add_argument('--loss', type=str, nargs=1, required=True,
                        choices=['mse', 'ce'], help='Sepcify the loss type.')
    parser.add_argument('--path', type=str, nargs=1,
                        required=True, help='Model and output data save path.')
    parser.add_argument('--alpha', type=float, nargs='+',
                        default=[0.005], help='Specify the learning rate.')
    parser.add_argument('--lambda', type=float, nargs='+',
                        default=[0], dest='reg', help='Specify the regularization parameter.')
    parser.add_argument('--epochs', type=int, nargs=1,
                        default=[5000], help='Training epoches.')
    parser.add_argument('--batch_size', type=int, nargs=1,
                        default=[500], help='Batch size. For SGD only.')
    parser.add_argument('--beta1', type=float, nargs='+',
                        default=[0.9], help='Specify the beta1 hyperparameter for Adam.')
    parser.add_argument('--beta2', type=float, nargs='+',
                        default=[0.999], help='Specify the beta2 hyperparameter for Adam.')
    parser.add_argument('--epsilon', type=float, nargs='+',
                        default=[1e-8], help='Specify the epsilon hyperparameter for Adam.')

    args = parser.parse_args()
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    X, y = preprocess_data(trainData, trainTarget)
    N, d = X.shape
    Xt, yt = preprocess_data(testData, testTarget)
    Xv, yv = preprocess_data(validData, validTarget)

    epochs = args.epochs[0]

    # Output path
    path = args.path[0]
    create_dir(path)

    batch_size = args.batch_size[0]
    optimizer = args.optimizer[0]
    loss_type = args.loss[0]

    if optimizer == 'gd':
        for alpha, reg in itertools.product(args.alpha, args.reg):
            params = {
                'alpha': alpha,
                'reg': reg,
                'batchsize': batch_size,
            }

            model_file, loss_file, time_file = make_filenames(
                path,
                [optimizer, loss_type],
                params,
            )

            with TimeThis(time_file, params):
                w, perf_logger = SGD(X, y, Xv, yv, Xt, yt,
                                    epochs, batch_size, optimizer, loss_type, params)

            # Save model and loss data
            save_model(model_file, w)
            perf_logger.save(loss_file)


    elif optimizer == 'adam':
        for alpha, reg, beta1, beta2, epsilon in itertools.product(
            args.alpha, args.reg, args.beta1, args.beta2, args.epsilon
        ):
            params = {
                'alpha': alpha,
                'reg': reg,
                'beta1': beta1,
                'beta2': beta2,
                'epsilon': epsilon,
                'batchsize': batch_size,
            }

            model_file, loss_file, time_file = make_filenames(
                path,
                [optimizer, loss_type],
                params,
            )

            with TimeThis(time_file, {'optimizer': optimizer, 'loss_type': loss_type, **params}):
                w, perf_logger = SGD(
                    X, y, Xv, yv, Xt, yt, epochs, batch_size, optimizer, loss_type, params)

            # Save model and loss data
            save_model(model_file, w)
            perf_logger.save(loss_file)
