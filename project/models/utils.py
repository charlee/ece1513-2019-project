import os
import sys
import csv
from io import StringIO

def create_dir(path):
    if os.path.exists(path):
        sys.stderr.write('Warning: output path exists!\n')
    else:
        os.makedirs(path)

class PerfLogger:

    def __init__(self, columns):
        self.columns = columns
        self.loss_data = []

    def append(self, epoch, data, print_log=False):
        d = [data.get(c, '') for c in self.columns]
        self.loss_data.append([epoch, *d])

        if print_log:
            msg = ['Epoch: %d' % epoch]
            for c in self.columns:
                msg.append('%s = %s' % (c, data[c]))
            print(', '.join(msg))

    def save(self, filename):
        s = StringIO()
        writer = csv.writer(s)
        if os.path.exists(filename):
            writer.writerows(self.loss_data)
            with open(filename, 'a') as f:
                f.write(s.getvalue())
        else:
            writer.writerow(self.columns)
            writer.writerows(self.loss_data)
            with open(filename, 'w') as f:
                f.write(s.getvalue())

        # Clear data
        self.loss_data = []
