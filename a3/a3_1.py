import os
import csv
import tensorflow as tf
import numpy as np
from common import save_data


class KMeans:

    def __init__(self):
        self.epoch = 0
        self.history = []

    def distance(self, X, Mu):
        """Distance function for K-means.
        :param X: N x D matrix (N observations and D dimensions)
        :param Mu: K x D matrix (K means and D dimensions)
        :returns: pairwise distance matrix (N x K)
        """
        # Reshape X to N x 1 x D to enable broadcasting
        X_reshaped = tf.reshape(X, (-1, 1, X.shape[-1]))

        # subtraction is performed on the last two dimensions of X_reshaped
        # X_reshaped(N stack, 1 x D) - Mu(K x D) => (N stack, K x D)
        # norm(N stack, K x D) => (N stack, K)
        return tf.reduce_sum(tf.square(X_reshaped - Mu), axis=-1)

    def build_graph(self, K, D):
        """Create the computation graph.
        :param K: number of clusters.
        :param D: number of dimensions.
        """
        tf.reset_default_graph()

        # X -> N x D
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')

        # Mu -> K x D
        self.Mu = tf.get_variable('Mu', shape=(K, D),
                                   initializer=tf.initializers.truncated_normal())

        # d2 -> N x K (pairwise distance matrix)
        d2 = self.distance(self.X, self.Mu)

        # min(N x K) => (N)
        min_d2 = tf.reduce_min(d2, axis=1)

        # loss = sum(N) => scalar
        self.loss = tf.reduce_sum(min_d2, axis=0, name='loss')
        self.opt_op = tf.train.AdamOptimizer(
            learning_rate=1e-2,
            beta1=0.9,
            beta2=0.99,
            epsilon=1e-5
        ).minimize(self.loss)

        self.predict_op = tf.argmin(d2, axis=1)

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, n_epochs, tol=0.001):
        last_loss = -99999

        for i in range(n_epochs):
            self.epoch += 1
            self.sess.run(self.opt_op, feed_dict={self.X: data})

            loss_value = self.compute_loss(data)
            print('epoch %s, loss=%s' % (self.epoch, loss_value))

            self.history.append(loss_value)

            if np.abs(loss_value - last_loss) < tol:
                print('tolerance reached, early stop at epoch %s' % self.epoch)
                break
            
            last_loss = loss_value
    
        return self.history

    def compute_loss(self, data):
        return self.sess.run(self.loss, feed_dict={self.X: data})

    def predict(self, data):
        """Predict.

        Returns predicted labels and cluster centers.
        """
        return self.sess.run([self.predict_op, self.Mu], feed_dict={self.X: data})


def run_kmeans(K, data, epochs=1000, tol=0.001, with_valid=False):

    N, D = data.shape

    if with_valid:
        valid_batch = int(N / 3.0)
        rnd_idx = np.arange(N)
        np.random.shuffle(rnd_idx)

        X_valid = data[rnd_idx[:valid_batch]]
        X_train = data[rnd_idx[valid_batch:]]

    else:
        X_train = data


    kmeans = KMeans()
    kmeans.build_graph(K, D)
    kmeans.init_session()
    history = kmeans.train(X_train, n_epochs=epochs, tol=tol)

    y_train, c_train = kmeans.predict(X_train)

    result = {
        'k': K,
        'history': history,
        'train': {
            'X': X_train,
            'y': y_train,
            'Mu': c_train,
        },
    }

    if with_valid:
        valid_loss = kmeans.compute_loss(X_valid)
        y_valid, c_valid = kmeans.predict(X_valid)

        result.update({
            'valid': {
                'X': X_valid,
                'y': y_valid,
                'Mu': c_valid,
                'loss': valid_loss,
            },
        })
    
    return result


if __name__ == '__main__':

    data = np.load('data2D.npy')

    # Remove comment to run each problem

    # Run 1.1
    result = run_kmeans(3, data)
    save_data(result, '1.1')

    # Run 1.2
    for k in range(1, 6):
        result = run_kmeans(k, data)
        save_data(result, '1.2', str(k))

    # Run 1.3
    for k in range(1, 6):
        result = run_kmeans(k, data, with_valid=True)
        save_data(result, '1.3', str(k))

