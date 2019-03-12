import os
import hashlib
import requests
import shutil
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

BASE_URL = 'http://ufldl.stanford.edu/housenumbers'
TRAIN_DATA = 'train_32x32.mat'
TEST_DATA = 'test_32x32.mat'
EXTRA_DATA= 'extra_32x32.mat'
ALL_DATA = {
    'train': TRAIN_DATA, 
    'test': TEST_DATA, 
    'extra': EXTRA_DATA,
}

MD5 = {
    'train_32x32.mat': 'e26dedcc434d2e4c54c9b2d4a06d8373',
    'test_32x32.mat': 'eb5a983be6a315427106f1b164d9cef3',
    'extra_32x32.mat': 'a93ce644f1a588dc4d68dda5feec44a7',
}


def show_log(s):
    print(s, end='')
    sys.stdout.flush()

class SVHN:

    def __init__(self):
        cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(cur_dir, '__data__')

    def ensure_data_dir(self):
        """Create data dir."""
        if not os.path.exists(self.data_dir):
            show_log('Data dir %s not exist, creating...' % self.data_dir)
            os.makedirs(self.data_dir)
            show_log('done.\n')

    def ensure_datafiles(self):
        """Download data files."""
        self.ensure_data_dir()

        for file in ALL_DATA.values():
            file_path = os.path.join(self.data_dir, file)

            if not os.path.exists(file_path):
                file_url = '%s/%s' % (BASE_URL, file)
                show_log('%s not found, downloading from %s...' % (file, file_url))
                r = requests.get(file_url, stream=True)
                if r.status_code == 200:
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    show_log('done.\n')
                else:
                    show_log('FAILED.\n')
                    return

    def load_data(self, t):
        """Decode mat file to npz."""
        self.ensure_datafiles()
        datafile = os.path.join(self.data_dir, ALL_DATA[t])
        data = sio.loadmat(datafile)
        return data['X'].transpose(3, 0, 1, 2), data['y']

    def visualize(self, X, y):
        """Visualize the dataset."""
        imidx = []
        labels = np.unique(y)

        for label in labels:
            for i in range(y.shape[0]):
                if y[i] == label:
                    imidx.append(i)
                    break

        for i, label in enumerate(labels):
            plt.subplot(2, 5, i+1)
            img = X[imidx[i]]
            plt.imshow(img)
            plt.xlabel('%s' % label)

        plt.tight_layout()
        plt.show()

