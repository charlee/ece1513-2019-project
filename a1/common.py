import os
import sys
import time
import numpy as np
import pandas as pd

def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

class PerfLogger:

    def __init__(self, columns):
        self.columns = columns
        self.loss_data = pd.DataFrame(columns=['epoch', *columns])

    def append(self, epoch, data, print_log=False):
        self.loss_data = self.loss_data.append({
            'epoch': epoch,
            **data,
        }, ignore_index=True)

        if print_log:
            msg = ['Epoch: %d' % epoch]
            for c in self.columns:
                msg.append('%s = %s' % (c, data[c]))
            print(', '.join(msg))

    def save(self, filename):
        self.loss_data.to_csv(filename, index=False)


################################################################################
# Common helpers
################################################################################

def preprocess_data(inputs, labels):
    d = np.prod(inputs.shape[1:])

    X = np.reshape(inputs.astype(np.float64), [-1, d])
    N = X.shape[0]
    x0 = np.empty([N, 1])
    x0.fill(1.)
    X = np.hstack([x0, X])

    y = np.reshape(labels.astype(np.float64), [-1])

    return X, y


def predict(w, X):
    y_ = (np.sign(np.matmul(X, w) - .5) + 1) / 2
    return y_


def accuracy(w, X, y):
    y_ = predict(w, X)
    return np.sum(y == y_) / len(y_)


def save_model(filename, w):
    w.dump(filename)


def save_loss(filename, data):
    np.savetxt(filename, data, delimiter=',')

def make_filenames(path, types, hyperparams):
    type_str = '_'.join(types)
    params_str = '_'.join('%s=%s' % (k, v) for k, v in hyperparams.items())

    return [os.path.join(path, tmpl % (type_str, params_str)) for tmpl in [
        'model~%s~%s.npy',
        'loss~%s~%s.csv',
        'time~%s~%s.csv',
    ]]

class TimeThis(object):
    def __init__(self, filename, params):
        self.filename = filename
        self.params = params
    
    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, type, value, traceback):
        self.time_end = time.time()
        log = '\n'.join('%s=%s' % (k, v) for k, v in self.params.items())
        log += '\n' + 'time=%f' % (self.time_end - self.time_start)

        open(self.filename, 'w').write(log)


def create_dir(path):
    if os.path.exists(path):
        sys.stderr.write('Warning: output path exists!\n')
    else:
        os.makedirs(path)