import numpy as np
import argparse
import time
import os
import pickle

from common import loadData, PerfLogger, convertOneHot, make_filenames, create_dir, shuffle, TimeThis


def relu(x):
    """ReLU function.
    :param x: A vector of scalar numbers.
    """
    return np.maximum(x, np.zeros(x.shape))

def d_relu(x):
    """Derivative of ReLU function.
    :param x: A vector of scalar numbers.
    """
    return (x > 0).astype(np.int64)


def softmax(x):
    """softmax function.
    :param x: A 2-D vector, shape: (N, K), N = # of data points
    """
    e = np.exp(x - np.max(x, axis=1)[:,None])
    s = np.sum(e, axis=1)[:,None]   # row-wise sum
    return e / s                    # Divide each elem with corresponding row


def computeLayer(X, W, b):
    return np.matmul(X, W) + b


def CE(target, prediction):
    """Cross entropy loss.
    CE = -1/N sum{n=1..N} sum{k=1..K} t_k^(n) * log(s_k^(n))
    :param target: N * K
    :param prediction: N * K
    """
    N = target.shape[0]
    return -np.sum(target * np.log(prediction)) / N


def gradCE(target, prediction):
    """Gradient of CE.
    """
    return -np.mean(target / prediction, axis=0)


def xavier(shape):
    """Return Xavier initialization for given shape.
    """
    var = 2 / (shape[0] + shape[1])
    return np.random.normal(0, np.sqrt(var), shape)

class NeuronNetwork:

    # Convensions:
    #   - Xn: output of layer n-1; especially, X is input layer
    #   - sn: input of layer n
    #   - Wn: weight
    #   - bn: bias
    #   - delta_n: back propagation message
    #   - dA_B: (partial) derivative of A with respect of B.
    # Layer number:
    #   - 1: hidden layer
    #   - 2: output layer

    def __init__(self):
        pass

    def init_params(self, input_layer, hidden_layer, output_layer):
        """Build the neuron network.
        :param input_layer: Input layer size.
        :param hidden_layer: Hidden layer size.
        :param output_layer: Output layer size.
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layer = hidden_layer

        self.W1 = xavier(shape=(input_layer, hidden_layer))
        self.b1 = np.random.normal(0, np.sqrt(1 / hidden_layer), size=(hidden_layer,))
        self.W2 = xavier(shape=(hidden_layer, output_layer))
        self.b2 = np.random.normal(0, np.sqrt(1 / output_layer), size=(output_layer,))

        self.wv2 = np.full(shape=self.W2.shape, fill_value=1e-5)
        self.wv1 = np.full(shape=self.W1.shape, fill_value=1e-5)
        self.bv2 = np.full(shape=self.b2.shape, fill_value=1e-5)
        self.bv1 = np.full(shape=self.b1.shape, fill_value=1e-5)

        self.epoch = 0

    def fprop(self, X, y):
        """Compute front propagation."""
        s1 = computeLayer(X, self.W1, self.b1)
        X1 = relu(s1)
        s2 = computeLayer(X1, self.W2, self.b2)
        X2 = softmax(s2)

        return s1, X1, s2, X2

    def loss_accuracy(self, X, y):
        _, _, s2, X2 = self.fprop(X, y)
        loss = CE(y, X2)
        accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1)) / y.shape[0]
        return loss, accuracy

    def bprop(self, X, y, s1, X1, s2, X2):
        """Compute back propagation."""
        # Back propagation message for layer 2.
        # (dL_s2) = (dL_X2) * (dX2_s2)
        #         = y - X2
        delta_2 = X2 - y

        # (dL_W2) = (dL_s2) * (ds2_W2)
        #         = (dL_s2) * X1
        # In matrix form, it should be matmul(X1 * dL_s2)
        dL_W2 = np.mean(np.matmul(X1[:,:,None], delta_2[:,None,:]), axis=0)
        dL_b2 = np.mean(delta_2, axis=0)

        # Back propagation message for layer 1.
        # (dL_s1) = (dL_X1) * (dX1_s1)
        #         = (dL_X1) * d_relu(s1).
        # (dL_X1j) = sum_k((dL_s2k) * (ds2k_dX1j)
        #          = sum_k((dL_s2k) * W2jk).
        # In matrix form, (dL_X1) = matmul((dL_s2), W2.T)
        delta_1 = np.matmul(delta_2, self.W2.T) * d_relu(s1)

        dL_W1 = np.mean(np.matmul(X[:,:,None], delta_1[:,None,:]), axis=0)
        dL_b1 = np.mean(delta_1, axis=0)

        return dL_W2, dL_b2, dL_W1, dL_b1


    def train_step(self, X, y, alpha=1e-5, gamma=0.99):
        """Train one step.
        :param X: input matrix.
        :param y: target vector, one-hot encoded.
        """

        # import ipdb; ipdb.set_trace()
        # Run forward propagation and compute the loss.
        s1, X1, s2, X2 = self.fprop(X, y)

        # Run back propagation and compute gradients for W and b.
        dL_W2, dL_b2, dL_W1, dL_b1 = self.bprop(X, y, s1, X1, s2, X2)

        # Update weights and biases
        self.wv2 = gamma * self.wv2 + alpha * dL_W2
        self.W2 -= self.wv2

        self.bv2 = gamma * self.bv2 + alpha * dL_b2
        self.b2 -= self.bv2

        self.wv1 = gamma * self.wv1 + alpha * dL_W1
        self.W1 -= self.wv1

        self.bv1 = gamma * self.bv1 + alpha * dL_b1
        self.b1 -= self.bv1
    
    def SGD_epoch(self, trainData, trainTarget, batch_size=100, alpha=1e-5, gamma=0.99):
        """Train one epoch with SGD."""
        data, target = shuffle(trainData, trainTarget)
        batch_count = int(np.ceil(trainData.shape[0] / batch_size))

        self.epoch += 1

        for step in range(batch_count):
            X = data[step * batch_size : (step + 1) * batch_size]
            y = target[step * batch_size : (step + 1) * batch_size]
            self.train_step(X, y, alpha, gamma)

    def save_model(self, model_file):
        """Save the model to file."""
        model_params = {
            'epoch': self.epoch,
            'W1': self.W1.tolist(),
            'W2': self.W2.tolist(),
            'b1': self.b1.tolist(),
            'b2': self.b2.tolist(),
            'wv1': self.wv1.tolist(),
            'wv2': self.wv2.tolist(),
            'bv1': self.bv1.tolist(),
            'bv2': self.bv2.tolist(),
            'input_layer': self.input_layer,
            'hidden_layer': self.hidden_layer,
            'output_layer': self.output_layer,
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model_params, f)

    @classmethod
    def load_model(cls, model_file):
        """Load the model from file."""
        with open(model_file, 'rb') as f:
            model_params = pickle.load(f)

        nn = NeuronNetwork()

        nn.input_layer = model_params['input_layer']
        nn.hidden_layer = model_params['hidden_layer']
        nn.output_layer = model_params['output_layer']

        nn.W1 = np.array(model_params['W1'])
        nn.W2 = np.array(model_params['W2'])
        nn.b1 = np.array(model_params['b1'])
        nn.b2 = np.array(model_params['b2'])
        nn.wv1 = np.array(model_params['wv1'])
        nn.wv2 = np.array(model_params['wv2'])
        nn.bv1 = np.array(model_params['bv1'])
        nn.bv2 = np.array(model_params['bv2'])

        nn.epoch = model_params['epoch']

        return nn


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs=1,
                        required=True, help='Model and output data save path.')
    parser.add_argument('--epochs', type=int, nargs=1,
                        default=[200], help='Training epoches.')
    parser.add_argument('--hidden', type=int, nargs=1,
                        default=[1000], help='Hidden layer size.')
    args = parser.parse_args()

    # Load and preprocess data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget1, validTarget1, testTarget1 = convertOneHot(trainTarget, validTarget, testTarget)
    trainData = np.reshape(trainData, (trainData.shape[0], -1))
    validData = np.reshape(validData, (validData.shape[0], -1))
    testData = np.reshape(testData, (testData.shape[0], -1))

    path = args.path[0]
    epochs = args.epochs[0]
    params = { 'hidden': args.hidden[0] }

    model_file, loss_file, time_file = make_filenames(path, ['nn'], params)
    perf_logger = PerfLogger([
        'train_loss', 'train_accuracy',
        'valid_loss', 'valid_accuracy',
        'test_loss', 'test_accuracy'
    ])

    model_loaded = False
    if os.path.exists(model_file):
        # Continue previous training process
        try:
            nn = NeuronNetwork.load_model(model_file)
            print('Saved model found (epoch=%s), continue training process.' % nn.epoch)
            model_loaded = True
        except:
            pass

    if not model_loaded:
        create_dir(path)
        nn = NeuronNetwork()
        nn.init_params(trainData.shape[1], params['hidden'], trainTarget1.shape[1])
        print('Saved model not found, start new training process.')

    with TimeThis(time_file) as tt:
        while nn.epoch < epochs:
            nn.SGD_epoch(trainData, trainTarget1, alpha=0.001, gamma=0.99)

            train_loss, train_accuracy = nn.loss_accuracy(trainData, trainTarget1)
            valid_loss, valid_accuracy = nn.loss_accuracy(validData, validTarget1)
            test_loss, test_accuracy = nn.loss_accuracy(testData, testTarget1)
        
            perf_logger.append(nn.epoch, {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
            }, print_log=True)

            # Save model and loss data
            tt.save_time()
            nn.save_model(model_file)
            perf_logger.save(loss_file)
