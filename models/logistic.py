from .base import Model
from .layers import Dense, Flatten

class LogisticRegression(Model):

    def build_graph(self, image_shape, n_classes, alpha=1e-6, l2=0.001):
        self._make_input(image_shape, n_classes)

        self.layers = [
            Flatten(),
            Dense(n_classes)
        ]

        X = self.fprop()

        self.make_softmax_loss(X, l2)
        self.make_predict(X)
        self.make_optimizer(alpha)