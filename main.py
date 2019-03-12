from dataset import SVHN

svhn = SVHN()

X, y = svhn.load_data('train')