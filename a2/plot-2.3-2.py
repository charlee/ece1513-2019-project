import pandas as pd
import matplotlib.pyplot as plt


df9 = pd.read_csv('./2.3-2/loss~cnn~l2=0.0_dropout=0.9.csv')
df75 = pd.read_csv('./2.3-2/loss~cnn~l2=0.0_dropout=0.75.csv')
df5 = pd.read_csv('./2.3-2/loss~cnn~l2=0.0_dropout=0.5.csv')


fig = plt.figure(figsize=(11, 4))

plt.subplot(1, 3, 1)
plt.plot(df9.epoch, df9.train_accuracy, label='p=0.9', c='k', linestyle=':')
plt.plot(df75.epoch, df75.train_accuracy, label='p=0.75', c='k', linestyle='-')
plt.plot(df5.epoch, df5.train_accuracy, label='p=0.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Train Accuracy')

plt.subplot(1, 3, 2)
plt.plot(df9.epoch, df9.valid_accuracy, label='p=0.9', c='k', linestyle=':')
plt.plot(df75.epoch, df75.valid_accuracy, label='p=0.75', c='k', linestyle='-')
plt.plot(df5.epoch, df5.valid_accuracy, label='p=0.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Valid Accuracy')

plt.subplot(1, 3, 3)
line1 = plt.plot(df9.epoch, df9.test_accuracy, label='p=0.9', c='k', linestyle=':')
line2 = plt.plot(df75.epoch, df75.test_accuracy, label='p=0.75', c='k', linestyle='-')
line3 = plt.plot(df5.epoch, df5.test_accuracy, label='p=0.5', c='k', linestyle='--')
plt.xlabel('epochs')
plt.title('Test Accuracy')

plt.suptitle(r'Train, Valid, and Test Accuracy of CNN w/ different dropout probability')

plt.figlegend([line1[0], line2[0], line3[0]], ['p=0.9', 'p=0.75', 'p=0.5'], loc='lower center', ncol=3)

fig.tight_layout(rect=[0, 0.06, 1, 0.95])

plt.savefig('./2.3-2/accuracy.png')
