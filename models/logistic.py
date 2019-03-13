from .base import Model

class LogisticRegression(Model):

    def build_graph(self, image_shape, n_classes):
        self._make_input(image_shape, n_classes)
