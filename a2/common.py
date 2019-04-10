import os
from io import StringIO
import sys
import time
import numpy as np
import pandas as pd


def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
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
        if os.path.exists(filename):
            s = StringIO()
            self.loss_data.to_csv(s, index=False, header=False)
            with open(filename, 'a') as f:
                f.write(s.getvalue())
        else:
            self.loss_data.to_csv(filename, index=False)

        # Clear data
        self.loss_data = self.loss_data.iloc[0:0]



def make_filenames(path, types, hyperparams):
    type_str = '_'.join(types)
    params_str = '_'.join('%s=%s' % (k, v) for k, v in hyperparams.items())

    return [os.path.join(path, tmpl % (type_str, params_str)) for tmpl in [
        'model~%s~%s.model',
        'loss~%s~%s.csv',
        'time~%s~%s.csv',
    ]]

class TimeThis(object):
    def __init__(self, filename):
        self.filename = filename
        self.elapsed_time = 0

        if os.path.exists(filename):
            line = open(filename).read()
            line = line.strip()
            if line:
                self.elapsed_time = float(line)
    
    def __enter__(self):
        self.time_start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.save_time()

    def save_time(self):
        self.time_end = time.time()
        log = '%s' % (self.time_end - self.time_start + self.elapsed_time)
        open(self.filename, 'w').write(log)


def create_dir(path):
    if os.path.exists(path):
        sys.stderr.write('Warning: output path exists!\n')
    else:
        os.makedirs(path)


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest