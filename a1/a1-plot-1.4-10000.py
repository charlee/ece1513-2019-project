import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt

lines = ['-', ':', '--', '-.']
linecycler = cycler(linestyle=lines)
matplotlib.rcParams['axes.prop_cycle'] = linecycler

df001 = pd.read_csv('a1-1.4-10000/loss~lr~alpha=0.005_reg=0.001.csv')
df1 = pd.read_csv('a1-1.4-10000/loss~lr~alpha=0.005_reg=0.1.csv')
df5 = pd.read_csv('a1-1.4-10000/loss~lr~alpha=0.005_reg=0.5.csv')

# plot train loss
plt.plot(df001[100:].epoch, df001[100:].train_loss, label=r'$\lambda=.001$')
plt.plot(df1[100:].epoch, df1[100:].train_loss, label=r'$\lambda=.1$')
plt.plot(df5[100:].epoch, df5[100:].train_loss, label=r'$\lambda=.5$')
plt.annotate(
    'epoch=%d\nloss=%.4f' % (df001.iloc[-1].epoch, df001.iloc[-1].train_loss),
    (df001.iloc[-1].epoch, df001.iloc[-1].train_loss),
    xytext=(df001.iloc[-1].epoch - 2000, df001.iloc[-1].train_loss + 0.0005),
)
plt.annotate('epoch=%d\nloss=%.4f' % (
    df1.iloc[-1].epoch, df1.iloc[-1].train_loss), (df1.iloc[-1].epoch, df1.iloc[-1].train_loss))
plt.annotate('epoch=%d\nloss=%.4f' % (
    df5.iloc[-1].epoch, df5.iloc[-1].train_loss), (df5.iloc[-1].epoch, df5.iloc[-1].train_loss))
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend()
plt.title(r'Train Loss comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/train_loss.png')
plt.close()

# plot test loss
plt.plot(df001[100:].epoch, df001[100:].test_loss, label=r'$\lambda=.001$')
plt.plot(df1[100:].epoch, df1[100:].test_loss, label=r'$\lambda=.1$')
plt.plot(df5[100:].epoch, df5[100:].test_loss, label=r'$\lambda=.5$')
plt.annotate(
    'epoch=%d\nloss=%.4f' % (df001.iloc[-1].epoch, df001.iloc[-1].test_loss),
    (df001.iloc[-1].epoch, df001.iloc[-1].test_loss),
    xytext=(df001.iloc[-1].epoch - 2000, df001.iloc[-1].test_loss + 0.0002),
)
plt.annotate('epoch=%d\nloss=%.4f' % (
    df1.iloc[-1].epoch, df1.iloc[-1].test_loss), (df1.iloc[-1].epoch, df1.iloc[-1].test_loss))
plt.annotate('epoch=%d\nloss=%.4f' % (
    df5.iloc[-1].epoch, df5.iloc[-1].test_loss), (df5.iloc[-1].epoch, df5.iloc[-1].test_loss))
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend()
plt.title(r'Test Loss comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/test_loss.png')

plt.close()

# plot valid loss
plt.plot(df001[100:].epoch, df001[100:].valid_loss, label=r'$\lambda=.001$')
plt.plot(df1[100:].epoch, df1[100:].valid_loss, label=r'$\lambda=.1$')
plt.plot(df5[100:].epoch, df5[100:].valid_loss, label=r'$\lambda=.5$')
plt.annotate(
    'epoch=%d\nloss=%.4f' % (df001.iloc[-1].epoch, df001.iloc[-1].valid_loss),
    (df001.iloc[-1].epoch, df001.iloc[-1].valid_loss),
    xytext=(df001.iloc[-1].epoch - 2000, df001.iloc[-1].valid_loss + 0.0002),
)
plt.annotate('epoch=%d\nloss=%.4f' % (
    df1.iloc[-1].epoch, df1.iloc[-1].valid_loss), (df1.iloc[-1].epoch, df1.iloc[-1].valid_loss))
plt.annotate('epoch=%d\nloss=%.4f' % (
    df5.iloc[-1].epoch, df5.iloc[-1].valid_loss), (df5.iloc[-1].epoch, df5.iloc[-1].valid_loss))
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.title(r'Validation Loss comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/valid_loss.png')

plt.close()


# plot train accuracy
plt.plot(df001.epoch, df001.train_accuracy, label=r'$\lambda=.001$')
plt.plot(df1.epoch, df1.train_accuracy, label=r'$\lambda=.1$')
plt.plot(df5.epoch, df5.train_accuracy, label=r'$\lambda=.5$')
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.legend()
plt.title(r'Train Accuracy comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/train_accuracy.png')
plt.close()

# plot test accuracy
plt.plot(df001.epoch, df001.test_accuracy, label=r'$\lambda=.001$')
plt.plot(df1.epoch, df1.test_accuracy, label=r'$\lambda=.1$')
plt.plot(df5.epoch, df5.test_accuracy, label=r'$\lambda=.5$')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title(r'Test Accuracy comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/test_accuracy.png')

plt.close()

# plot valid accuracy
plt.plot(df001.epoch, df001.valid_accuracy, label=r'$\lambda=.001$')
plt.plot(df1.epoch, df1.valid_accuracy, label=r'$\lambda=.1$')
plt.plot(df5.epoch, df5.valid_accuracy, label=r'$\lambda=.5$')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title(r'Validation Accuracy comparison on different $\lambda$ ($\alpha=.005$)')
plt.savefig('a1-1.4-10000/valid_accuracy.png')

plt.close()