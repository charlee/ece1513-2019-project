
class Model:

    def __init__(self, logdir):
        """Constructor.
        :param logdir: The directory used to save the model.
        """
        np.random.seed()
        create_dir(logdir)
        self.logdir = logdir
        self.epoch = 0

    def shuffle(self, X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def next_batch(self, X, y, batch_size):
        shuffled_X, shuffled_y = self.shuffle(X, y)
        n_batch = int(np.ceil(len(X) / batch_size))
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            yield shuffled_X[start:end], shuffled_y[start:end]
