Assignment 2
===============

## Scripts

- `a2.py`: Python script for problem 1.
- `a2-tf.py`: Python script for problem 2.

## a2.py

### Paramters

- `--path`: Data output dir.
- `--epochs`: Epochs. Default = 200.
- `--hidden`: Hidden layer size. Default = 1000.

## a2-tf.py

### Parameters

- `--logdir`: Data output dir.
- `--l2`: L2 regularization parameter. Default = 0 (no regularziation).
- `--dropout`: Dropout layer probability (keep probability). Default = 1 (no dropout).
- `--tensorboard`: Output tensorboard data or not. Default = False. 

## Examples

```
# 1.3
python a2.py --path 1.3 --hidden 1000

# 1.4
python a2.py --path 1.4-100 --hidden 100
python a2.py --path 1.4-500 --hidden 500
python a2.py --path 1.4-2000 --hidden 2000

# 2.2
python a2-tf.py --logdir 2.2

# 2.3-1
python a2-tf.py --logdir 2.3-1 --l2 0.01
python a2-tf.py --logdir 2.3-1 --l2 0.1
python a2-tf.py --logdir 2.3-1 --l2 0.5

# 2.3-2
python a2-tf.py --logdir 2.3-2 --dropout 0.9
python a2-tf.py --logdir 2.3-2 --dropout 0.75
python a2-tf.py --logdir 2.3-2 --dropout 0.5
```

## Plots

- plot-1.3.py
- plot-1.4.py
- plot-2.2.py
- plot-2.3-1.py
- plot-2.3-2.py
