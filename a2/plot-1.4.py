import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('./1.4-100/loss~nn~hidden=100.csv')
df5 = pd.read_csv('./1.4-500/loss~nn~hidden=500.csv')
df2 = pd.read_csv('./1.4-2000/loss~nn~hidden=2000.csv')


fig = plt.figure(figsize=(8.5, 4))

plt.subplot(1, 2, 1)
plt.plot(df1.epoch, df1.train_accuracy, label='hidden=100', c='k', linestyle='-')
plt.plot(df5.epoch, df5.train_accuracy, label='hidden=500', c='k', linestyle=':')
plt.plot(df2.epoch, df2.train_accuracy, label='hidden=2000', c='k', linestyle='--')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy')

plt.subplot(1, 2, 2)
plt.plot(df1.epoch, df1.test_accuracy, label='hidden=100', c='k', linestyle='-')
plt.plot(df5.epoch, df5.test_accuracy, label='hidden=500', c='k', linestyle=':')
plt.plot(df2.epoch, df2.test_accuracy, label='hidden=2000', c='k', linestyle='--')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('test accuracy')
plt.title('Test Accuracy')

plt.suptitle(r'Train and Test Accuracy of NN w/ different hidden units')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('./1.4-100/accuracy.png')