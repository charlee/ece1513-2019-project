import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from common import loadData, PerfLogger, shuffle, make_filenames, create_dir
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class CNN:

    def __init__(self, logdir):
        """Constructor.
        :param logdir: The directory used to save the model.
        """
        create_dir(logdir)
        self.logdir = logdir

    def build_graph(self, image_size=28, num_of_class=10, conv_params=(3, 32, 1),
                    pool_size=2, alpha=1e-4):
        """Build a convolutional neuron network.

        :param image_size: The width/height of the input image. Image must be a square
                           with size of (image_size, image_size).
        :param num_of_class: Number of output classes.
        :param conv_params: A 3-element tuple, specifying (kernel_size, num_of_filters,
                            stride) for conv layer.
        :param pool_size: Size of max_pool layer.
        :param alpha: Learning rate.
        """
        with tf.name_scope('input'):
            # 1. Input layer
            with tf.name_scope('X'):
                self.X = tf.placeholder(tf.float32, shape=(None, image_size, image_size),
                                        name='X')
                X_reshaped = tf.reshape(self.X, shape=(-1, image_size, image_size, 1),
                                        name='X_reshaped')

            # Target
            with tf.name_scope('Y'):
                self.y = tf.placeholder(tf.int64, shape=(None,), name='y')
                y_onehot = tf.one_hot(self.y, num_of_class)

        with tf.name_scope('conv_pool'):

            # 2. Conv layer
            with tf.name_scope('conv'):
                kernel_size, num_of_filters, stride = conv_params
                W_conv = tf.get_variable('W_conv', dtype=tf.float32,
                                         shape=(kernel_size, kernel_size, 1, num_of_filters),
                                         initializer=tf.contrib.layers.xavier_initializer())
                b_conv = tf.get_variable('b_conv', shape=(num_of_filters,),
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(X_reshaped, W_conv, strides=(1, stride, stride, 1),
                                    padding='SAME') + b_conv

            # 3. ReLU activation
            h_conv = tf.nn.relu(conv)

            # 4. Bath normalization
            with tf.name_scope('batch_norm'):
                h_mean, h_var = tf.nn.moments(h_conv, axes=[0])
                batch_norm = tf.nn.batch_normalization(h_conv, h_mean, h_var, None, None, 1e-9)

            # 5. Max pooling
            with tf.name_scope('pool'):
                h_pool = tf.nn.max_pool(batch_norm, ksize=[1, pool_size, pool_size, 1],
                                        strides=[1, pool_size, pool_size, 1],
                                        padding='SAME', name='h_pool')

            # 6. Flatten
            with tf.name_scope('flatten'):
                feature_count = int(image_size * image_size / pool_size / pool_size *
                                    num_of_filters)
                h_flat = tf.reshape(h_pool, shape=[-1, feature_count], name='h_flat')

        with tf.name_scope('fc1'):

            # 7. FC1 layer
            with tf.name_scope('fc1'):
                fc_size = image_size * image_size
                W_fc1 = tf.get_variable('W_fc1', shape=(feature_count, fc_size), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                b_fc1 = tf.get_variable('b_fc1', shape=(fc_size,), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                h_fc1 = tf.matmul(h_flat, W_fc1) + b_fc1

            # Dropout layer
            with tf.name_scope('dropout'):
                self.p_dropout = tf.placeholder(tf.float32, shape=(), name='p_dropout')
                h_fc1 = tf.nn.dropout(h_fc1, keep_prob=self.p_dropout)

            # 8. ReLU
            h_fc1 = tf.nn.relu(h_fc1)

        with tf.name_scope('fc2'):
            # 9. FC2 layer
            W_fc2 = tf.get_variable('W_fc2', shape=(fc_size, num_of_class), dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = tf.get_variable('b_fc2', shape=(num_of_class,), dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

        with tf.name_scope('output'):

            # 10. Softmax
            h_softmax = tf.nn.softmax(h_fc2)

        # 11. Cross Entropy Loss
        with tf.name_scope('optimize'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_onehot, logits=h_fc2))
            
            # L2 regularization
            self.l2_lambda = tf.placeholder(tf.float32, shape=(), name='l2_lambda')
            reg = self.l2_lambda * (tf.nn.l2_loss(W_conv) + tf.nn.l2_loss(W_fc1) +
                                    tf.nn.l2_loss(W_fc2))
            self.loss = self.loss + reg

            # Optimizer
            self.opt_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)

        # Predict
        with tf.name_scope('predict'):
            self.predict = tf.argmax(h_softmax, 1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predict, self.y), tf.float32)
            )

    def set_data(self, trainData, trainTarget, validData, validTarget, testData, testTarget):
        """Set the data to be used in training process.
        """
        self.trainData = trainData
        self.trainTarget = trainTarget
        self.validData = validData
        self.validTarget = validTarget
        self.testData = testData
        self.testTarget = testTarget

    def init_session(self):
        """Start a new training session.
        """
        tf.set_random_seed(421)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.epoch = 0

    def train(self, l2=0.0, dropout=1.0, batch_size=32, epochs=50, with_tensorboard=False):
        """Run the training process. Call init_session() before calling this.
        :param l2: L2 regularization parameter.
        :param dropout: Dropout probability.
        :param batch_size: Batch size.
        :param with_tensorboard: Whether output data for displaying in tensorboard.
        """
        if with_tensorboard:
            # Writers for tensorboard
            self.train_writer = tf.summary.FileWriter(os.path.join(self.logdir, 'train'), self.sess.graph)
            self.valid_writer = tf.summary.FileWriter(os.path.join(self.logdir, 'valid'), self.sess.graph)
            self.test_writer = tf.summary.FileWriter(os.path.join(self.logdir, 'test'), self.sess.graph)

        # Although we have tensorboard, we still want to reuse the PerfLogger
        # defined in previous answer to provide consistent performance logs.
        params = {'l2': l2, 'dropout': dropout}
        _, self.loss_file, _ = make_filenames(self.logdir, ['cnn'], params)
        self.perf_logger = PerfLogger([
            'train_loss', 'train_accuracy',
            'valid_loss', 'valid_accuracy',
            'test_loss', 'test_accuracy',
        ])

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged_summary = tf.summary.merge_all()

        while self.epoch < epochs:
            self.run_epoch(l2=l2, dropout=dropout, batch_size=batch_size, 
                           with_tensorboard=with_tensorboard)

    def run_epoch(self, l2=0.0, dropout=1.0, batch_size=32, with_tensorboard=False):
        """Run one epoch of the training process.
        :param l2: L2 regularization parameter.
        :param dropout: Dropout probability.
        :param batch_size: Batch size.
        :param with_tensorboard: Whether output data for displaying in tensorboard.
        """

        self.epoch += 1

        N = self.trainData.shape[0]
        shuffled_X, shuffled_y = shuffle(self.trainData, self.trainTarget)

        batch_count = int(np.ceil(N / batch_size))
        for i in range(batch_count):
            # Train one batch
            X_input = shuffled_X[batch_size * i:batch_size * (i+1)]
            y_input = shuffled_y[batch_size * i:batch_size * (i+1)]

            feed_dict = {
                self.X: X_input,
                self.y: y_input,
                self.l2_lambda: l2,
                self.p_dropout: dropout,
            }

            self.sess.run(self.opt_op, feed_dict=feed_dict)

        # Compute train loss / accuracy
        train_loss, train_accuracy, train_summary = self.sess.run(
            [self.loss, self.accuracy, self.merged_summary],
            feed_dict=feed_dict)

        # Compute valid loss / accuracy
        feed_dict = {
            self.X: validData,
            self.y: validTarget,
            self.l2_lambda: l2,
            self.p_dropout: 1.0,
        }
        valid_loss, valid_accuracy, valid_summary = self.sess.run(
            [self.loss, self.accuracy, self.merged_summary],
            feed_dict=feed_dict)

        # Compute test loss / accuracy
        feed_dict = {
            self.X: testData,
            self.y: testTarget,
            self.l2_lambda: l2,
            self.p_dropout: 1.0,
        }
        test_loss, test_accuracy, test_summary = self.sess.run(
            [self.loss, self.accuracy, self.merged_summary],
            feed_dict=feed_dict)

        # Save to tensorboard
        if with_tensorboard:
            self.train_writer.add_summary(train_summary, self.epoch)
            self.valid_writer.add_summary(valid_summary, self.epoch)
            self.test_writer.add_summary(test_summary, self.epoch)

        # Save to PerfLogger
        self.perf_logger.append(self.epoch, {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'valid_loss': valid_loss,
            'valid_accuracy': valid_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
        }, print_log=True)
        self.perf_logger.save(self.loss_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, nargs=1, required=True, help='Logdir.')
    parser.add_argument('--l2', type=float, nargs=1, default=[0.0], help='L2 regression parameter.')
    parser.add_argument('--dropout', type=float, nargs=1, default=[1.0], help='Dropout probability.')
    parser.add_argument('--tensorboard', action='store_true', help='Output tensorboard data.')
    args = parser.parse_args()

    cnn = CNN(args.logdir[0])
    cnn.build_graph()

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget = trainTarget.astype(np.int64)
    validTarget = validTarget.astype(np.int64)
    testTarget = testTarget.astype(np.int64)

    cnn.set_data(trainData, trainTarget, validData, validTarget, testData, testTarget)
    cnn.init_session()

    cnn.train(l2=args.l2[0], dropout=args.dropout[0], epochs=50, with_tensorboard=args.tensorboard)
