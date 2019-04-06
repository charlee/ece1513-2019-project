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
        self.checkpoint_file = os.path.join(self.logdir, 'model.ckpt')

    def _shuffle(self, X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def set_phase(self, phase):
        self.phase = phase

    def _next_batch(self, X, y, batch_size, shuffle=True):
        if shuffle:
            shuffled_X, shuffled_y = self._shuffle(X, y)
        else:
            shuffled_X, shuffled_y = X, y

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

    def fprop(self):
        X = self.X
        for layer in self.layers:
            X = layer.fprop(X)

        return X

    def make_softmax_loss(self, X, l2=None):
        with tf.name_scope('loss'):
            # Loss w/ L2 regularization
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_onehot, logits=X))
            
            if l2 is not None:
                # L2 regularization
                weights = [layer.get_W() for layer in self.layers]
                reg = tf.add_n([tf.nn.l2_loss(W) for W in weights if W is not None])
                self.loss = loss + l2 * reg

    def make_optimizer(self, learning_rate):
        with tf.name_scope('optimize'):
            self.opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def make_predict(self, X):
        with tf.name_scope('predict'):
            self.output = tf.nn.softmax(X)

            self.predict = tf.argmax(self.output, axis=1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predict, self.y), tf.float32)
            )

    def build_graph(self, image_shape, n_classes, layers, alpha=1e-4, l2=0.001):
        """Build the compute graph.
        :param image_shape: The width/height of the input image. Image must be a square
                           with size of (image_size, image_size).
        :param n_classes: Number of output classes.
        :param conv_params: A 3-element tuple, specifying (kernel_size, num_of_filters,
                            stride) for conv layer.
        :param pool_size: Size of max_pool layer.
        :param alpha: Learning rate.

        Must set the following tensors:

        self.loss: Loss tensor
        self.output: Raw predict output
        self.opt_op: Optimizer
        self.predict: Prediction
        self.accuracy: Accuracy
        """
        self._make_input(image_shape, n_classes)
        self.layers = layers

        X = self.fprop()

        self.make_softmax_loss(X, l2)
        self.make_predict(X)
        self.make_optimizer(alpha)

    def set_data(self, trainData, trainTarget, testData, testTarget):
        """Set the data to be used in training process.
        """
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.testData = testData
        self.testTarget = testTarget

    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.checkpoint_file)
        print('Model saved to %s.' % self.checkpoint_file)

    def restore_model(self):
        if os.path.exists('%s.index' % self.checkpoint_file):
            saver = tf.train.Saver()
            saver.restore(self.sess, self.checkpoint_file)
            print('Model restored from %s.' % self.checkpoint_file)

    def init_session(self):
        """Start a new training session.
        """
        # Create a tensor to store epoch so that global epoch can be saved in checkpoint
        self.epoch = tf.get_variable('epoch', shape=(), initializer=tf.zeros_initializer)
        self.inc_epoch = self.epoch.assign(self.epoch + 1)

        tf.set_random_seed(421)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.restore_model()

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

            # Save model every 100 epoch
            if (i+1) % 100 == 0:
                self.save_model()
        
        # Save model at the end of training
        self.save_model()

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

        if epoch % 10 == 0:
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

    def run_predict(self, batch_size=100):
        y_ = []
        for X, y in self._next_batch(self.testData, self.testTarget, batch_size):
            feed_dict = self.get_feed_dict(X, y, 'test', shuffle=False)
            y_.append(self.sess.run(
                self.predict,
                feed_dict=self.get_feed_dict(X_batch, y_batch, 'test')
            ))

        return np.vstack(y_)
