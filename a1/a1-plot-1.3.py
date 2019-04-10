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

df1 = pd.read_csv('a1-1.3/loss~lr~alpha=0.0001_reg=0.csv')
df2 = pd.read_csv('a1-1.3/loss~lr~alpha=0.001_reg=0.csv')
df3 = pd.read_csv('a1-1.3/loss~lr~alpha=0.005_reg=0.csv')

plt.subplot(1, 3, 1)
plt.plot(df1.epoch, df1.train_loss, label=r'$\alpha=0.0001$')
plt.plot(df2.epoch, df2.train_loss, label=r'$\alpha=0.001$')
plt.plot(df3.epoch, df3.train_loss, label=r'$\alpha=0.005$')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(df1.epoch, df1.valid_loss, label=r'$\alpha=0.0001$')
plt.plot(df2.epoch, df2.valid_loss, label=r'$\alpha=0.001$')
plt.plot(df3.epoch, df3.valid_loss, label=r'$\alpha=0.005$')
plt.xlabel('Epochs')
plt.ylabel('Valid Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(df1.epoch, df1.test_loss, label=r'$\alpha=0.0001$')
plt.plot(df2.epoch, df2.test_loss, label=r'$\alpha=0.001$')
plt.plot(df3.epoch, df3.test_loss, label=r'$\alpha=0.005$')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend()

plt.suptitle(r'Linear Regression Loss $\lambda=0$')

plt.show()
