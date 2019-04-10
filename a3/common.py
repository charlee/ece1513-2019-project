import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def save_data(result, path, suffix='1'):
    if not os.path.exists(path):
        os.makedirs(path)

    fn = lambda n: os.path.join(path, n % suffix)

    pickle.dump(result, open(fn('data-%s.pkl'), 'wb'))


def mvn_pdf(x, mu, var):
    """Single version, used to plot contour."""
    D = len(mu)
    cov_inv = np.diag([1/var] * D)
    cov_det = var ** D
    dist = x - mu

    return (2 * np.pi) ** (-.5 * D) * cov_det ** (-.5) * np.exp(-.5 * dist.T @ cov_inv @ dist)

def plot_contour(X, Mu, Var):
    # MVN contour
    x1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    x2 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    z = np.zeros((len(x1), len(x2), len(Mu)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            for k in range(len(Mu)):
                z[j,i,k] = mvn_pdf((x1[i], x2[j]), Mu[k], Var[k])
    for k in range(len(Mu)):
        plt.contour(x1, x2, z[:,:,k], cmap='coolwarm', linewidths=1)


def plot_scatter(k, loss, X, y, Mu, Var=None, phase='train'):

    use_pca = False
    if X.shape[1] > 2:
        use_pca = True
        pca = PCA(2).fit(X)
        X_ = pca.transform(X)
        Mu_ = pca.transform(Mu)
    else:
        X_ = X
        Mu_ = Mu

    plt.scatter(X_[:,0], X_[:,1], c=y, cmap='tab10')
    plt.scatter(Mu_[:,0], Mu_[:,1], c='r', marker='x')

    if use_pca:
        plt.xlabel('pc1 ({}%)'.format(int(pca.explained_variance_ratio_[0] * 100)))
        plt.ylabel('pc2 ({}%)'.format(int(pca.explained_variance_ratio_[1] * 100)))
    else:
        plt.xlabel('x1')
        plt.ylabel('x2')

    if Var is not None and not use_pca:
        plot_contour(X, Mu, Var)

    plt.title('%s clustering, K=%s, loss=%s' % (phase, k, loss))
    plt.tight_layout()


def plot_result(path, suffix='1', figsize=(5, 4)):

    fn = lambda n: os.path.join(path, n % suffix)

    result = pickle.load(open(fn('data-%s.pkl'), 'rb'))

    # train loss
    history = result['history']
    epochs = np.arange(len(history)) + 1

    plt.figure(figsize=figsize)
    plt.plot(epochs, np.array(history) / 1000, label='train loss')
    plt.annotate(
        'Converge: \nepoch=%s\nloss=%s' % (len(history), history[-1]),
        (len(history), history[-1] / 1000),
        xytext=(len(history) / 2, (history[0] + history[-1]) / 2000),
        arrowprops={'arrowstyle': '->'},
    )
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss/1000')
    plt.title('train loss, K=%s' % (result['k'],))
    plt.tight_layout()
    plt.savefig(fn('loss-%s.png'))

    # train loss csv
    with open(fn('loss-%s.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'train_loss'])
        csv_writer.writerows(zip(epochs, history))

    # Train Cluster
    train = result['train']
    X, y, Mu = train['X'], train['y'], train['Mu']
    Var = train.get('Var', None)

    plt.figure(figsize=figsize)
    plot_scatter(result['k'], history[-1], X, y, Mu, Var, phase='train')
    plt.savefig(fn('train-cluster-%s.png'))

    # Valid Cluster
    if 'valid' in result:
        valid = result['valid']
        X, y, Mu = valid['X'], valid['y'], valid['Mu']
        Var = valid.get('Var', None)

        plt.figure(figsize=figsize)
        plot_scatter(result['k'], valid['loss'], X, y, Mu, Var, phase='valid')
        plt.savefig(fn('valid-cluster-%s.png'))
