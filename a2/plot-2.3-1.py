import pandas as pd
import matplotlib.pyplot as plt


df01 = pd.read_csv('./2.3-1/loss~cnn~l2=0.01_dropout=1.0.csv')
df1 = pd.read_csv('./2.3-1/loss~cnn~l2=0.1_dropout=1.0.csv')
df5 = pd.read_csv('./2.3-1/loss~cnn~l2=0.5_dropout=1.0.csv')


fig = plt.figure(figsize=(11, 4))

plt.subplot(1, 3, 1)
plt.plot(df01.epoch, df01.train_accuracy, label='l2=.01', c='k', linestyle='-')
plt.plot(df1.epoch, df1.train_accuracy, label='l2=.1', c='k', linestyle=':')
plt.plot(df5.epoch, df5.train_accuracy, label='l2=.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Train Accuracy')

plt.subplot(1, 3, 2)
plt.plot(df01.epoch, df01.valid_accuracy, label='l2=.01', c='k', linestyle='-')
plt.plot(df1.epoch, df1.valid_accuracy, label='l2=.1', c='k', linestyle=':')
plt.plot(df5.epoch, df5.valid_accuracy, label='l2=.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Valid Accuracy')

plt.subplot(1, 3, 3)
line1 = plt.plot(df01.epoch, df01.test_accuracy, label='l2=.01', c='k', linestyle='-')
line2 = plt.plot(df1.epoch, df1.test_accuracy, label='l2=.1', c='k', linestyle=':')
line3 = plt.plot(df5.epoch, df5.test_accuracy, label='l2=.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Test Accuracy')

plt.suptitle(r'Train, Valid, and Test Accuracy of CNN w/ different L2 regularization')

plt.figlegend([line1[0], line2[0], line3[0]], ['l2=0.01', 'l2=0.1', 'l2=0.5'], loc='lower center', ncol=3)

fig.tight_layout(rect=[0, 0.06, 1, 0.95])

plt.savefig('./2.3-1/accuracy.png')
