import tensorflow as tf
import numpy as np

from .base import Model


class Layer:
    def __init__(self, **kwargs):
        self.phase = 'train'
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_model(self, model):
        self.model = model

    def fprop(self, X):
        pass

    def make_var(self, name, shape):
        return tf.get_variable(
            name, dtype=tf.float32, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer()
        )

    def make_W(self, shape):
        self.W = self.make_var('W_%s' % self.name, shape)

    def make_b(self, shape):
        self.b = self.make_var('b_%s' % self.name, shape)

    def get_W(self):
        return getattr(self, 'W', None)

    def get_b(self):
        return getattr(self, 'b', None)

    def get_feed_dict(self):
        return {}


class ConvLayer(Layer):
    def __init__(self, kernel_size, n_filters, stride, name='conv'):
        super().__init__(kernel_size=kernel_size, n_filters=n_filters, stride=stride, name=name)

    def fprop(self, X):
        self.make_W([self.kernel_size, self.kernel_size, X.shape[-1], self.n_filters])
        self.make_b([self.n_filters])

        conv = tf.nn.conv2d(X, self.W, strides=(1, self.stride, self.stride, 1), padding='SAME')
        return tf.add(conv, self.b, name=self.name)

class PoolLayer(Layer):
    def __init__(self, pool_size, name='pool'):
        super().__init__(pool_size=pool_size, name=name)

    def fprop(self, X):
        shape = [1, self.pool_size, self.pool_size, 1]
        return tf.nn.max_pool(X, ksize=shape, strides=shape, padding='SAME', name=self.name)

class BatchNormLayer(Layer):
    def __init__(self, name='batch_norm'):
        super().__init__(name=name)

    def fprop(self, X):
        mean, var = tf.nn.moments(X, axes=[0])
        return tf.nn.batch_normalization(
               X, mean, var, None, None, 1e-9, name=self.name)


class FlattenLayer(Layer):
    def __init__(self, name='flatten'):
        super().__init__(name=name)

    def fprop(self, X):
        output_size = np.prod(X.shape[1:])
        return tf.reshape(X, shape=(-1, output_size), name=self.name)


class FCLayer(Layer):
    def __init__(self, output_size, name='fc'):
        super().__init__(output_size=output_size, name=name)

    def fprop(self, X):
        input_size = X.shape[-1]
        self.make_W([input_size, self.output_size])
        self.make_b([self.output_size])

        return tf.add(tf.matmul(X, self.W), self.b)


class DropoutLayer(Layer):
    def __init__(self, keep_prob, name='dropout'):
        super().__init__(keep_prob=keep_prob, name=name)
        self.keep_prop_tensor = tf.placeholder(tf.float32, shape=())
    
    def fprop(self, X):
        return tf.nn.dropout(X, keep_prob=self.keep_prop_tensor)

    def get_feed_dict(self):
        if self.model.phase == 'train':
            return {self.keep_prop_tensor: self.keep_prob}
        else:
            return {self.keep_prop_tensor: 1.0}


class ReLULayer(Layer):
    def __init__(self, name='relu'):
        super().__init__(name=name)

    def fprop(self, X):
        return tf.nn.relu(X, name=self.name)



class SoftmaxLayer(Layer):
    def __init__(self, name='relu'):
        super().__init__(name=name)

    def fprop(self, X):
        return tf.nn.softmax(X, name=self.name)


class CNN(Model):

    def build_graph(self, image_shape, n_classes, layers, alpha=1e-4, l2=0.001):
        """Build a convolutional neuron network.

        :param image_shape: The width/height of the input image. Image must be a square
                           with size of (image_size, image_size).
        :param n_classes: Number of output classes.
        :param conv_params: A 3-element tuple, specifying (kernel_size, num_of_filters,
                            stride) for conv layer.
        :param pool_size: Size of max_pool layer.
        :param alpha: Learning rate.
        """
        self._make_input(image_shape, n_classes)
        self.layers = layers

        X = self.X
        for layer in layers:
            layer.set_model(self)
            X = layer.fprop(X)

        # Loss w/ L2 regularization
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.y_onehot, logits=X))
        
        # L2 regularization
        weights = [layer.get_W() for layer in self.layers]
        reg = tf.add_n([tf.nn.l2_loss(W) for W in weights if W is not None])
        self.loss = loss + l2 * reg

        self.output = tf.nn.softmax(X)

        self.opt_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)

        self.predict = tf.argmax(self.output, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predict, self.y), tf.float32)
        )
