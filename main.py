from dataset import SVHN

svhn = SVHN()

X, y = svhn.load_data('train')
svhn.visualize(X, y)