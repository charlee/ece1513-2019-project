import tensorflow as tf
import random
import string
import numpy as np

class Layer:
    def __init__(self, **kwargs):
        self.hash = ''.join(random.choice(string.ascii_lowercase) for _ in range(4))
        self.phase = 'train'
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_model(self, model):
        self.model = model

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

    def get_feed_dict(self):
        return {}


class Conv2D(Layer):
    def __init__(self, n_filters, kernel_size, strides=(1, 1), activation=None):
        super().__init__(
            kernel_size=kernel_size,
            n_filters=n_filters,
            strides=strides,
            activation=activation,
        )

    def fprop(self, X):
        scope = 'conv_%s' % self.hash
        with tf.name_scope(scope):
            self.make_W([*self.kernel_size, X.shape[-1], self.n_filters], scope=scope)
            self.make_b([self.n_filters], scope=scope)
            strides = (1, *self.strides, 1)

            out = tf.nn.conv2d(X, self.W, strides=strides, padding='SAME') + self.b
            if self.activation == 'relu':
                out = tf.nn.relu(out)

            return out

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        super().__init__(pool_size=pool_size)

    def fprop(self, X):
        with tf.name_scope('pool_%s' % self.hash):
            shape = [1, *self.pool_size, 1]
            return tf.nn.max_pool(X, ksize=shape, strides=shape, padding='SAME')

class BatchNorm(Layer):
    def fprop(self, X):
        with tf.name_scope('batch_norm_%s' % self.hash):
            mean, var = tf.nn.moments(X, axes=[0])
            return tf.nn.batch_normalization(
                X, mean, var, None, None, 1e-9)


class Flatten(Layer):
    def fprop(self, X):
        with tf.name_scope('flatten_%s' % self.hash):
            output_size = np.prod(X.shape[1:])
            return tf.reshape(X, shape=(-1, output_size))


class Dense(Layer):
    def __init__(self, output_size, activation=None):
        super().__init__(output_size=output_size, activation=activation)

    def fprop(self, X):
        scope = 'fc_%s' % self.hash
        with tf.name_scope(scope):
            input_size = X.shape[-1]
            self.make_W([input_size, self.output_size], scope=scope)
            self.make_b([self.output_size], scope=scope)

            out = tf.matmul(X, self.W) + self.b

            if self.activation == 'relu':
                out = tf.nn.relu(out)
            
            return out


class Dropout(Layer):
    def __init__(self, keep_prob):
        super().__init__(keep_prob=keep_prob)
        self.keep_prop_tensor = tf.placeholder(tf.float32, shape=())
    
    def fprop(self, X):
        with tf.name_scope('dropout_%s' % self.hash):
            return tf.nn.dropout(X, keep_prob=self.keep_prop_tensor)

    def get_feed_dict(self):
        if self.model.phase == 'train':
            return {self.keep_prop_tensor: self.keep_prob}
        else:
            return {self.keep_prop_tensor: 1.0}


class ReLU(Layer):
    def fprop(self, X):
        with tf.name_scope('relu_%s' % self.hash):
            return tf.nn.relu(X)

