import tensorflow as tf

from .base import Model


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

        with tf.name_scope('loss'):
            # Loss w/ L2 regularization
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_onehot, logits=X))
            
            # L2 regularization
            weights = [layer.get_W() for layer in self.layers]
            reg = tf.add_n([tf.nn.l2_loss(W) for W in weights if W is not None])
            self.loss = loss + l2 * reg

        with tf.name_scope('optimize'):
            self.opt_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)

        with tf.name_scope('predict'):
            self.output = tf.nn.softmax(X)

            self.predict = tf.argmax(self.output, axis=1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predict, self.y), tf.float32)
            )
