import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./2.2/loss~cnn~l2=0.0_dropout=1.0.csv')

fig = plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)

plt.plot(df.epoch, df.train_loss, label='train loss', c='k', linestyle='-')
plt.plot(df.epoch, df.valid_loss, label='valid loss', c='k', linestyle=':')
plt.plot(df.epoch, df.test_loss, label='test loss', c='k', linestyle='--')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(df.epoch, df.train_accuracy, label='train accuracy', c='k', linestyle='-')
plt.plot(df.epoch, df.valid_accuracy, label='valid accuracy', c='k', linestyle=':')
plt.plot(df.epoch, df.test_accuracy, label='test accuracy', c='k', linestyle='--')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy')

plt.suptitle(r'Loss and Accuracy of CNN (l2=0, dropout=1)')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('./2.2/loss.png')
