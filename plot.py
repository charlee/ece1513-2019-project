import pandas as pd
import matplotlib.pyplot as plt


for m in ['lr', 'nn', 'cnn', 'simplenet']:
    df = pd.read_csv('__model-%s__/loss.csv' % m)
    
    plt.figure(figsize=(4,3))
    plt.plot(df.index, df.train_loss, label='train')
    plt.plot(df.index, df.test_loss, label='test')
    plt.legend()
    plt.title('Train / test loss (model=%s)' % m)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('__model-%s__/%s-loss.png' % (m, m))

    plt.figure(figsize=(4,3))
    plt.plot(df.index, df.train_accuracy, label='train')
    plt.plot(df.index, df.test_accuracy, label='test')
    plt.legend()
    plt.title('Train / test error (model=%s)' % m)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.tight_layout()
    plt.savefig('__model-%s__/%s-error.png' % (m, m))