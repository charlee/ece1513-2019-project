import time
import os
import sys
import argparse
import itertools
import numpy as np

from common import (
    loadData, PerfLogger, predict, accuracy, save_loss,
    save_model, preprocess_data, create_dir, make_filenames,
    TimeThis
)


################################################################################
# Common math functions
################################################################################

def norm_sqr(v):
    '''Calculate norm square.'''
    return np.inner(v, v)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

################################################################################
# Linear regression
################################################################################


def MSE(w, X, y, reg):
    # reshape each data point to 1-d vector
    N = X.shape[0]

    # MSE
    ld = norm_sqr(np.matmul(X, w) - y) / 2 / N
    # Use only original w for regularization
    lw = norm_sqr(w[1:]) * reg / 2

    return ld + lw


def gradMSE(w, X, y, reg):
    # Gradient
    N = X.shape[0]
    return (np.matmul(np.matmul(X.T, X), w) - np.matmul(X.T, y)) / N + reg * np.concatenate([[0], w[1:]])

################################################################################
# Logistic regression
################################################################################


def crossEntropyLoss(w, X, y, reg):
    N, d = X.shape
    total = 0
    for i in range(N):
        z = np.matmul(w, X[i])
        total += -z * y[i] + np.log(1 + np.exp(z))
    ld = total / N
    # Use only original w for regularization
    lw = norm_sqr(w[1:]) * reg / 2

    return ld + lw


def gradCE(w, X, y, reg):
    """Gradient for cross entropy."""
    N, d = X.shape
    grad = np.zeros(d)
    for i in range(N):
        z = np.matmul(w, X[i])
        grad = grad + (sigmoid(z) - y[i]) * X[i]
    return grad / N + reg * np.concatenate([[0], w[1:]])


################################################################################
# Optimizers
################################################################################


def grad_descent_step(w, X, y, alpha, reg, lossType):
    """Run one step for Linear regression with Gradient Descent."""
    if lossType == 'ce':
        grad = gradCE(w, X, y, reg)
        loss = crossEntropyLoss(w, X, y, reg)
    elif lossType == 'mse':
        grad = gradMSE(w, X, y, reg)
        loss = MSE(w, X, y, reg)
    else:
        sys.stderr.write('Error: wrong lossType! (lossType=%s)' % lossType)
        sys.exit()

    w -= grad * alpha

    return w, loss


def full_grad_descent(X, y, Xv, yv, Xt, yt, alpha, reg, lossType, epochs, err_tolerance=1e-7):
    """Run Batched Gradient Descent.
    :param X: training input
    :param y: training labels
    :param Xv: validation input
    :param yv: validation labels 
    :param Xt: test input
    :param yt: test labels 
    :param alpha: learning rate
    :parma reg: Regularization param(lambda)
    :param lossType: 'ce' or 'mse', loss function type
    :param epochs: training epochs
    :returns: w, loss data
    """
    w = np.zeros(d)

    perf_logger = PerfLogger([
        'train_loss', 'train_accuracy',
        'valid_loss', 'valid_accuracy',
        'test_loss', 'test_accuracy'
    ])

    if lossType == 'ce':
        loss_func = crossEntropyLoss
    elif lossType == 'mse':
        loss_func = MSE
    else:
        sys.stderr.write('Invalid lossType: %s' % lossType)
        sys.exit()

    last_loss = -1
    stop = False

    for i in range(1, epochs + 1):
        w, loss = grad_descent_step(w, X, y, alpha, reg, lossType=lossType)

        if last_loss != -1 and abs(last_loss - loss) < err_tolerance:
            stop = True

        # To accelerate learning, we only compute valid/test loss every 10 epochs
        if i == 1 or i % 10 == 0 or i == epochs or stop:
            perf_logger.append(i, {
                'train_loss': loss,
                'train_accuracy': accuracy(w, X, y),
                'valid_loss': loss_func(w, Xv, yv, reg),
                'valid_accuracy': accuracy(w, Xv, yv),
                'test_loss': loss_func(w, Xt, yt, reg),
                'test_accuracy': accuracy(w, Xt, yt),
            }, print_log=True)

        if stop:
            break

        last_loss = loss

    return w, perf_logger


def linear_regression_normal_equation(X, y, reg):
    """Use analytics formular to find optimum solution for linear regression.
    W* = (X^T * X + \lambda * I)^-1 * X^T * Y
    """
    d = X.shape[1]
    return np.matmul(
        np.matmul(
            np.linalg.inv(
                np.matmul(X.T, X) + reg * np.eye(d)
            ),
            X.T
        ),
        y
    )


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, nargs=1, required=True,
                        choices=['lr', 'lrne', 'log', 'sgd'], help='Specify the algorithm.')
    parser.add_argument('--alpha', type=float, nargs='+',
                        default=[0.005], help='Specify the learning rate.')
    parser.add_argument('--lambda', type=float, nargs='+',
                        default=[0], dest='reg', help='Specify the regularization parameter.')
    parser.add_argument('--path', type=str, nargs=1,
                        required=True, help='Model and output data save path.')
    parser.add_argument('--epochs', type=int, nargs=1,
                        default=[5000], help='Training epoches.')
    args = parser.parse_args()

    # Load and preprocess data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    X, y = preprocess_data(trainData, trainTarget)
    N, d = X.shape
    Xt, yt = preprocess_data(testData, testTarget)
    Xv, yv = preprocess_data(validData, validTarget)

    alg = args.alg[0]
    epochs = args.epochs[0]

    # Output path
    path = args.path[0]
    create_dir(path)

    if alg == 'lr' or alg == 'log':

        if alg == 'log':
            lossType = 'ce'
        else:
            lossType = 'mse'

        # Do gradient descent for each alpha-lambda combination
        for alpha, reg in itertools.product(args.alpha, args.reg):
            params = {
                'alpha': alpha,
                'reg': reg,
            }

            model_file, loss_file, time_file = make_filenames(path, [alg], params)

            with TimeThis(time_file, params):
                w, perf_logger = full_grad_descent(
                    X, y, Xv, yv, Xt, yt, alpha, reg, lossType, epochs)

            # Save model and loss data
            save_model(model_file, w)
            perf_logger.save(loss_file)


    elif alg == 'lrne':
        for reg in args.reg:
            params = {
                'reg': reg,
            }

            model_file, loss_file, time_file = make_filenames(path, [alg], params)

            perf_logger = PerfLogger([
                'train_loss', 'train_accuracy',
                'valid_loss', 'valid_accuracy',
                'test_loss', 'test_accuracy'
            ])

            time_start = time.time()

            with TimeThis(time_file, params):
                w = linear_regression_normal_equation(X, y, reg)

            perf_logger.append(1, {
                'train_loss': MSE(w, X, y, reg),
                'train_accuracy': accuracy(w, X, y),
                'valid_loss': MSE(w, Xv, yv, reg),
                'valid_accuracy': accuracy(w, Xv, yv),
                'test_loss': MSE(w, Xt, yt, reg),
                'test_accuracy': accuracy(w, Xt, yt),
            })

            # Save model and loss data
            save_model(model_file, w)
            perf_logger.save(loss_file)
