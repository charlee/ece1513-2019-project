import tensorflow as tf
from itertools import count
import random
import string
import numpy as np

class Layer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fprop(self, X):
        pass

    def make_var(self, shape, name):
        return tf.get_variable(
            name, dtype=tf.float32, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer()
        )

    def make_W(self, shape, scope):
        self.W = self.make_var(shape, '%s/W' % scope)

    def make_b(self, shape, scope):
        self.b = self.make_var(shape, '%s/b' % scope)

    def get_W(self):
        return getattr(self, 'W', None)

    def get_b(self):
        return getattr(self, 'b', None)

    def get_feed_dict(self, phase='train'):
        return {}


class Conv2D(Layer):

    counter = count(1)

    def __init__(self, n_filters, kernel_size, strides=(1, 1), activation=None, batch_normal=None):
        """
        :param batch_normal: Specify mean and var for batch normal. (-1, -1) will calculate automaticlaly using momentum on X.

        """
        super().__init__(
            kernel_size=kernel_size,
            n_filters=n_filters,
            strides=strides,
            activation=activation,
            batch_normal=batch_normal,
        )

    def fprop(self, X):
        scope = 'conv_%s' % next(self.counter)
        with tf.name_scope(scope):
            self.make_W([*self.kernel_size, X.shape[-1], self.n_filters], scope=scope)
            self.make_b([self.n_filters], scope=scope)
            strides = (1, *self.strides, 1)

            out = tf.nn.conv2d(X, self.W, strides=strides, padding='SAME') + self.b

            if self.batch_normal is not None:
                mean, avg = self.batch_normal
                if mean == -1 and avg == -1:
                    mean, var = tf.nn.moments(out, axes=[0])
                out = tf.nn.batch_normalization(out, mean, var, None, None, 1e-9)

            if self.activation == 'relu':
                out = tf.nn.relu(out)

            return out

class MaxPooling2D(Layer):

    counter = count(1)

    def __init__(self, pool_size=(2, 2)):
        super().__init__(pool_size=pool_size)

    def fprop(self, X):
        with tf.name_scope('pool_%s' % next(self.counter)):
            shape = [1, *self.pool_size, 1]
            return tf.nn.max_pool(X, ksize=shape, strides=shape, padding='SAME')

class BatchNorm(Layer):

    counter = count(1)

    def fprop(self, X):
        with tf.name_scope('batch_norm_%s' % next(self.counter)):
            mean, var = tf.nn.moments(X, axes=[0])
            return tf.nn.batch_normalization(
                X, mean, var, None, None, 1e-9)


class Flatten(Layer):

    counter = count(1)

    def fprop(self, X):
        with tf.name_scope('flatten_%s' % next(self.counter)):
            output_size = np.prod(X.shape[1:])
            return tf.reshape(X, shape=(-1, output_size))


class Dense(Layer):
    counter = count(1)
    def __init__(self, output_size, activation=None):
        super().__init__(output_size=output_size, activation=activation)

    def fprop(self, X):
        scope = 'fc_%s' % next(self.counter)
        with tf.name_scope(scope):
            input_size = X.shape[-1]
            self.make_W([input_size, self.output_size], scope=scope)
            self.make_b([self.output_size], scope=scope)

            out = tf.matmul(X, self.W) + self.b

            if self.activation == 'relu':
                out = tf.nn.relu(out)
            
            return out


class Dropout(Layer):
    counter = count(1)
    def __init__(self, keep_prob):
        super().__init__(keep_prob=keep_prob)
        self.keep_prop_tensor = tf.placeholder(tf.float32, shape=())
    
    def fprop(self, X):
        with tf.name_scope('dropout_%s' % next(self.counter)):
            return tf.nn.dropout(X, keep_prob=self.keep_prop_tensor)

    def get_feed_dict(self, phase='train'):
        if phase == 'train':
            return {self.keep_prop_tensor: self.keep_prob}
        else:
            return {self.keep_prop_tensor: 1.0}
