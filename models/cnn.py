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
            X = layer.fprop(X)

        self.make_softmax_loss(X, l2)
        self.make_predict(X)
        self.make_optimizer(alpha)
