import os
import sys

def create_dir(path):
    if os.path.exists(path):
        sys.stderr.write('Warning: output path exists!\n')
    else:
        os.makedirs(path)

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
