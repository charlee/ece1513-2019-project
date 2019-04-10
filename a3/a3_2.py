import os
import csv
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helper import logsoftmax, reduce_logsumexp
from common import save_data


def next_batch(n, batch_size):
    batch_count = int(np.ceil(n / batch_size))
    idx = np.arange(n)
    np.random.shuffle(idx)

    for batch in range(batch_count):
        yield idx[batch*batch_size:(batch+1)*batch_size]


class MoG:

    def __init__(self):
        self.epoch = 0
        self.history = []

    def log_pdf(self, X, Mu, Var):
        """
        Compute the log PDF using multivariate Gaussian distribution.

        N(x; mu, cov) = (2pi)^{-.5d} * |cov|^{-.5} * exp(-.5 * (x-mu).T @ cov_inv @ (x-mu))
            =>
        log(N(x; mu, cov))
            = -.5d * log(2pi) - .5 * log|cov| - .5 * (x-mu).T @ cov_inv @ (x-mu)
            = -.5 * (d * log(2pi) + log|cov| + (x-mu).T @ cov_inv @ (x-mu))

        since cov is diagonal matrix with var on diagonal,
            |cov| = var ^ D
            cov_inv = I * (1 / var)
        thus
        log(N(x; mu, cov)) = -.5 * (d * log(2pi) + D * log(var) + (x-mu).T @ cov_inv @ (x-mu))
                           = -.5 * (d * log(2pi) + D * log(var) + (x-mu).T @ I @ (x-mu)) / var
                           = -.5 * (d * log(2pi) + D * log(var) + (x-mu).T @ (x-mu)) / var

        :param X: (N x D)
        :param Mu: (K x D)
        :param Var: (K,)
        """
        # Dimension
        K, D = Mu.shape.as_list()

        # We need to compute (x - mu)^T @ (x - mu) for all k in K, n in N,
        # where (x - mu) is with shape of (D, 1).
        # Thus the actual multiplication should be done with
        # dst_T(N, K, 1, D) @ dist(N, K, D, 1) => (N, K, 1, 1)
        dist = X[:, None, :, None] - Mu[None, :, :, None]   # (N x K x D x 1)
        dist_T = tf.transpose(dist, (0, 1, 3, 2))           # (N x K x 1 x D)

        # => (N x K)
        return -.5 * (
            D * np.log(2 * np.pi) +
            D * tf.log(Var) +
            tf.reshape(dist_T @ dist, shape=(-1, K)) / Var
        )

    def log_proba(self, X, Mu, Phi, Psi):
        """Compute the log probability of the cluster variable z given the data vector x.

        P(z=k|x) = P(x, z=k) / sum_j(P(x, z=j))
                 = P(z=k)P(x|z=k) / sum_j(P(z=j)P(x|z=j))

        where P(z=j) = w_j, s.t. w_j > 0 && sum_j(w_j) = 1,
              P(x|z=j) = N(x; mu_j, var_j)
        =>
        P(z=k|x) = w_k * N(x; mu_k, var_k) / sum_j(w_j * N(x; mu_j, var_j))
        =>
        log(P(z=k|x)) = log(w_k) + log(N(x; mu_k, var_k)) -
                        log(sum_j(w_j * N(x; mu_j, var_j)))

        Denote w_k = exp(psi_k) / sum_j(exp(psi_j)), var_k = exp(phi_k), then

        - log(N(x; mu_k, var_k)) = logpdf(x, mu_k, exp(phi_k))
        - log(w_k) = log(exp(psi_k)) - log(sum_j(exp(psi_j)))
                   = psi_k - logsumexp_K(Psi),
        - log(W) = [log(w_1) .. log(w_k)]
                 = [psi_1 - logsumexp_K(Psi), ..., psi_k - logsumexp_K(Psi)]
                 = Psi - logsumexp_K(Psi)
                 = logsoftmax(Psi)
        - log(sum_j(w_j * N(x; mu_j, var_j)))
             = log(sum_j(exp(log(w_j * N(x; mu_j, var_j)))))
             = logsumexp_K(log(W) + logpdf(x, Mu, Var))

        Thus
        log(P(z=k|x)) = log(w_k) + logpdf(x, mu_k, var_k) -
                        logsumexp_K(log(W) + logpdf(x, Mu, Var))

        log(P(z|x)) = [log(w_1) + logpdf(x, mu_1, var_1) - logsumexp_K, ...]
                    = [log(w_1) + logpdf(x, mu_1, var_1), ...] -
                      logsumexp_K(log(W) + logpdf(x, Mu, Var))
                    = (log(W) + logpdf(x, Mu, Var)) - logsumexp_K(log(W) + logpdf(x, Mu, Var))

        :param X: data matrix, (N x D)
        :param Mu: cluster center matrix, (K x D)
        :param Phi: (K,)
        :param Psi: (K,)
        :return (N x K), for each data point x_n, the log probability that it belongs to cluster k
        """
        N, D = X.shape
        K = Mu.shape[0]

        logpdf = self.log_pdf(X, Mu, tf.exp(Phi))       # (N x K)
        logw = logsoftmax(Psi)                          # (K,)

        logw_pdf = logpdf + logw                # (N x K)
        lse = reduce_logsumexp(logw_pdf, reduction_indices=1, keep_dims=True)   # (N, 1)

        return logw_pdf - lse                   # (N x K)


    def nll_loss(self, X, Mu, Phi, Psi):
        """Negative log likelyhood loss.

        We need to maximize P(X):
        P(X) = prod_n(P(x_n))
             = prod_n(sum_k(P(z_n = k) * P(x_n | z_n = k)))
             = prod_n(sum_k(w_k * N(x_n; mu_k, var_k)))

        =>
        -log(P(X)) = -log(prod_n(...))
                   = -sum_n(log(sum_k(w_k * N(x_n; mu_k, var_k))))
                   = -sum_n(log(sum_k(exp(log(w_k * N(x_n; mu_k, var_k))))))
                   = -sum_n(logsumexp_K(log(W * N(x_n; Mu, Var))))
                   = -sum_n(logsumexp_K(log(W) + logpdf(x_n, Mu, Var)))

        Denote w_k = exp(psi_k) / sum_j(exp(psi_j)), var_k = exp(phi_k), then
        - log(w_k) = log(exp(psi_k)) - log(sum_j(exp(psi_j)))
                   = psi_k - logsumexp_K(Psi)
        - log(W) = log([w_1 .. w_k])
                 = [log(w_1) .. log(w_k)]
                 = [psi_1 - logsumexp_K(Psi) .. psi_k - logsumexp_K(Psi)]
                 = [psi_1 .. psi_k] - logsumexp_K(Psi)
                 = Psi - logsumexp_K(Psi)
                 = logsoftmax(Psi)
        - logpdf(x_n, Mu, Var) = logpdf(x_n, Mu, exp(Phi))

        =>
        -log(P(X)) = -sum_n(logsumexp_K(logsoftmax(Psi) + logpdf(x_n, Mu, exp(Phi))))

        :param X: (N x D)
        :param Mu: (K x D)
        :param Psi: (K,)
        :param Phi: (K,)
        """
        logpdf = self.log_pdf(X, Mu, tf.exp(Phi))       # (N x K)
        lsm = logsoftmax(Psi)                           # (K,)

        return -tf.reduce_sum(
            reduce_logsumexp(
                (lsm + logpdf),             # (N, K)
                reduction_indices=1,        # along K dimension
            ),                              # => (N,)
        )                                   # => ()


    def build_graph(self, K, D, mu_init=(0.0, 1.0)):
        """
        :param var: Variance of the Gaussian distribution on each dimension
        """

        tf.reset_default_graph()

        # X -> N x D
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')

        # Mu -> K x D
        self.Mu = tf.get_variable(
            'Mu', shape=(K, D), initializer=tf.initializers.truncated_normal(*mu_init))
        
        self.Psi = tf.get_variable(
            'Psi', shape=(K,), initializer=tf.initializers.truncated_normal())
        
        self.Phi = tf.get_variable(
            'Phi', shape=(K,), initializer=tf.initializers.truncated_normal())

        self.loss = self.nll_loss(self.X, self.Mu, self.Phi, self.Psi)

        self.opt_op = tf.train.AdamOptimizer(
            learning_rate=1e-2,
            beta1=0.9,
            beta2=0.99,
            epsilon=1e-5
        ).minimize(self.loss)
        
        self.predict_op = self.log_proba(self.X, self.Mu, self.Phi, self.Psi)

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def compute_loss(self, data, batch_size=-1):
        if batch_size > 0:
            loss = 0
            for batch_idx in next_batch(len(data), batch_size):
                loss += self.sess.run(self.loss, feed_dict={self.X: data[batch_idx]}) * len(batch_idx)
            return loss / len(data)
        else:
            return self.sess.run(self.loss, feed_dict={self.X: data})

    def train(self, data, n_epochs, tol=0.001, batch_size=-1):
        last_loss = -99999

        for i in range(n_epochs):
            self.epoch += 1

            if batch_size > 0:
                for batch_idx in next_batch(len(data), batch_size):
                    self.sess.run(self.opt_op, feed_dict={self.X: data[batch_idx]})

            else:
                self.sess.run(self.opt_op, feed_dict={self.X: data})

            loss_value = self.compute_loss(data, batch_size=batch_size)
            print('epoch %s, loss=%s' % (self.epoch, loss_value))

            self.history.append(loss_value)

            if np.abs(loss_value - last_loss) < tol:
                print('tolerance reached, early stop at epoch %s' % self.epoch)
                break
            
            last_loss = loss_value
        
        Mu, Var, Phi, Psi = self.sess.run([self.Mu, tf.exp(self.Phi), self.Phi, self.Psi])
        print('Mu = %s, Var = %s, Phi = %s, Psi = %s' % (Mu, Var, Phi, Psi))
    
        return self.history

    def predict(self, data, batch_size=-1):
        """Predict.

        Returns predicted labels and cluster centers.
        """
        if batch_size > 0:
            preds = []
            Mus = []
            Vars = [] 
            for batch_idx in next_batch(len(data), batch_size):
                pred, Mu, Var = self.sess.run(
                    [self.predict_op, self.Mu, tf.exp(self.Phi)],
                    feed_dict={self.X: data[batch_idx]}
                )
                preds.append(pred)
                Mus.append(Mu)
                Vars.append(Var)
            return np.vstack(preds), np.vstack(Mus), np.vstack(Vars)
        else:
            return self.sess.run(
                [self.predict_op, self.Mu, tf.exp(self.Phi)],
                feed_dict={self.X: data}
            )


def run_mog(K, data, epochs=1500, tol=0.001, batch_size=-1, with_valid=False):

    N, D = data.shape

    if with_valid:
        valid_batch = int(N / 3.0)
        rnd_idx = np.arange(N)
        np.random.shuffle(rnd_idx)

        X_valid = data[rnd_idx[:valid_batch]]
        X_train = data[rnd_idx[valid_batch:]]

    else:
        X_train = data


    mog = MoG()
    mog.build_graph(K, D)
    mog.init_session()
    history = mog.train(X_train, n_epochs=epochs, batch_size=batch_size, tol=tol)

    y_train, Mu_train, Var_train = mog.predict(X_train, batch_size=batch_size)

    result = {
        'k': K,
        'history': history,
        'train': {
            'X': X_train,
            'prob': y_train,
            'y': np.argmax(y_train, axis=1),
            'Mu': Mu_train,
            'Var': Var_train,
        },
    }

    if with_valid:
        valid_loss = mog.compute_loss(X_valid)
        y_valid, Mu_valid, Var_valid = mog.predict(X_valid, batch_size=batch_size)

        result.update({
            'valid': {
                'X': X_valid,
                'prob': y_valid,
                'y': np.argmax(y_valid, axis=1),
                'Mu': Mu_valid,
                'Var': Var_valid,
                'loss': valid_loss,
            },
        })
    
    return result



if __name__ == '__main__':

    data = np.load('data2D.npy')
    data = data.astype(np.float32)

    # Run 2.2.1
    result = run_mog(3, data)
    save_data(result, '2.2.1')

    # # Run 2.2.2
    for k in range(1, 6):
        result = run_mog(k, data, tol=1e-8, epochs=3000, with_valid=True)
        save_data(result, '2.2.2', str(k))
