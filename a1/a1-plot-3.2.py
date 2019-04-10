import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt

lines = ['-', ':', '--', '-.']
linecycler = cycler(linestyle=lines)
matplotlib.rcParams['axes.prop_cycle'] = linecycler

df = pd.read_csv('a1-3.2/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=1e-08_batchsize=500.csv')

plt.subplot(1, 2, 1)
plt.plot(df.epoch, df.train_loss, label='train')
plt.plot(df.epoch, df.valid_loss, label='valid')
plt.plot(df.epoch, df.test_loss, label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df.epoch, df.train_accuracy, label='train')
plt.plot(df.epoch, df.valid_accuracy, label='valid')
plt.plot(df.epoch, df.test_accuracy, label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.suptitle(r'SGD w/ MSE loss & AdamOptimizer, batch_size=500, $\alpha=0.001, \lambda=0$')

plt.show()