import os
import numpy as np
import tensorflow as tf

from .utils import create_dir, PerfLogger

class Model:

    def __init__(self, logdir):
        """Constructor.
        :param logdir: The directory used to save the model.
        """
        np.random.seed()
        create_dir(logdir)
        self.logdir = logdir
        self.loss_file = os.path.join(self.logdir, 'loss.csv')
        self.epoch = 0
        self.phase = 'train'

    def _shuffle(self, X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def set_phase(self, phase):
        self.phase = phase

    def _next_batch(self, X, y, batch_size):
        shuffled_X, shuffled_y = self._shuffle(X, y)
        n_batch = int(np.ceil(len(X) / batch_size))
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            yield shuffled_X[start:end], shuffled_y[start:end]

    def _make_input(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.X = tf.placeholder(tf.float32, shape=(None, *input_shape), name='X')
        self.y = tf.placeholder(tf.int64, shape=(None,))
        self.y_onehot = tf.one_hot(self.y, n_classes)

    def build_graph(self, *args, **kwargs):
        """Build the compute graph.
        Must set the following tensors:

        self.loss: Loss tensor
        self.output: Raw predict output
        self.opt_op: Optimizer
        self.predict: Prediction
        self.accuracy: Accuracy
        """
        pass

    def set_data(self, trainData, trainTarget, testData, testTarget):
        """Set the data to be used in training process.
        """
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.testData = testData
        self.testTarget = testTarget

    def init_session(self):
        """Start a new training session.
        """
        # Create a tensor to store epoch so that global epoch can be saved in checkpoint
        self.epoch = tf.get_variable('epoch', shape=(), initializer=tf.zeros_initializer)
        self.inc_epoch = self.epoch.assign(self.epoch + 1)

        tf.set_random_seed(421)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_feed_dict(self, X, y, phase='train'):
        feed = {}
        for layer in self.layers:
            feed.update(layer.get_feed_dict(phase))
        feed.update({
            self.X: X,
            self.y: y,
        })
        return feed

    def train(self, batch_size=32, epochs=50):
        """Run the training process. Call init_session() before calling this.
        :param l2: L2 regularization parameter.
        :param dropout: Dropout probability.
        :param batch_size: Batch size.
        :param with_tensorboard: Whether output data for displaying in tensorboard.
        """
        self.perf_logger = PerfLogger([
            'train_loss', 'train_accuracy',
            'test_loss', 'test_accuracy',
        ])

        self.train_writer = tf.summary.FileWriter(os.path.join(self.logdir, 'train'), self.sess.graph)

        for i in range(epochs):
            self.run_epoch(batch_size=batch_size)

    def compute_loss_accuracy(self, X, y, batch_size=100):
        total_loss = 0
        total_accuracy = 0
        for X_batch, y_batch in self._next_batch(X, y, batch_size):
            loss, accuracy = self.sess.run(
                [self.loss, self.accuracy],
                feed_dict=self.get_feed_dict(X_batch, y_batch, 'test')
            )

            n = X_batch.shape[0]
            total_loss += loss * n
            total_accuracy += accuracy * n
        
        return total_loss / X.shape[0], total_accuracy / X.shape[0]


    def run_epoch(self, batch_size=100):
        """Run one epoch of the training process.
        :param batch_size: Batch size.
        """

        epoch = self.sess.run(self.inc_epoch)

        for X, y in self._next_batch(self.trainData, self.trainTarget, batch_size):

            feed_dict = self.get_feed_dict(X, y, 'train')
            self.sess.run(self.opt_op, feed_dict=feed_dict)

        if epoch % 1 == 0:
            # Compute train / test loss / accuracy
            train_loss, train_accuracy = self.compute_loss_accuracy(self.trainData, self.trainTarget)
            test_loss, test_accuracy = self.compute_loss_accuracy(self.testData, self.testTarget)

            # Save to PerfLogger
            self.perf_logger.append(epoch, {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
            }, print_log=True)
            self.perf_logger.save(self.loss_file)