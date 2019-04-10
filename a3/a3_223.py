import os
import csv
import numpy as np

from a3_1 import run_kmeans
from a3_2 import run_mog
from common import save_data



if __name__ == '__main__':
    data = np.load('data100D.npy')

    # Run 2.2.3
    for k in [5, 10, 15, 20, 30]:
        result = run_kmeans(k, data, epochs=1000, tol=1e-6)
        save_data(result, '2.2.3', 'kmeans-%s' % str(k))
    
    for k in [5, 10, 15, 20, 30]:
        result = run_mog(k, data, epochs=1000, tol=1e-8)
        save_data(result, '2.2.3', 'mog-%s' % str(k))
