from .cnn import CNN
from .layers import Dense, Flatten

class LogisticRegression(CNN):

    def build_graph(self, image_shape, n_classes, alpha=1e-6, l2=0.001):
        super().build_graph(
            image_shape,
            n_classes,
            layers=[
                Flatten(),
                Dense(n_classes)
            ],
            alpha=alpha,
            l2=l2,
        )