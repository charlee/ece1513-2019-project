Assignment 1
================


## Scripts

- `a1.py`: Python script, for problem 1 & 2.
- `a1-tf.py`: Script with tensorflow, for problem 3.
- `a1-plot.py`: Plotting script. This script accept multiple `--filename` and tries to plot give files onto one plot.
- `a1-plot-*.py`: Plotting script for individual problems.


## a1a.py

### Parameters

- `a1.py`
  - `alg`: Choose algorithm. REQUIRED. Valid values are:
    - `lr`: Linear regression, for 1.3~1.4
    - `lrne`: Linear regression with normal equation, for 1.5
    - `log`: Logistic regression with cross entropy loss.
  - `path`: Data and model output path. REQUIRED.
  - `alpha`: Learning rate. Not available for `lr-ne`. Multiple values can be used with the form of `--alpha 0.005 0.001 0.0001`. Default value is `0.005`.
  - `lambda`: Regularization parameter. Multiple values can be used. Default is `0`.
  - `epochs`: Training epochs. Default is `5000`.
- `a1-tf.py`:
  - `optimizer`: `gd` = GradientDescentOptimizer, `adam` = AdamOptimizer.
  - `loss`: `mse` or `ce`.
  - `path`: Data and model output path. REQUIRED.
  - `alpha`: Learning rate. Not available for `lr-ne`. Multiple values can be used with the form of `--alpha 0.005 0.001 0.0001`. Default value is `0.005`.
  - `lambda`: Regularization parameter. Multiple values can be used. Default is `0`.
  - `epochs`: Training epochs. Default is `5000`.
  - `batch_size`: Batch size for minibatch.
  - `beta1`: beta1 param for Adam.
  - `beta2`: beta2 param for Adam.
  - `epsilon`: epsilon param for Adam.

Example:
```bash
# Run linear regression with alpha=0.005 and lambda=[0.5, 0.1, 0.001]
$ python a1.py --alg=lr --path=a1-1.4 --alpha 0.005 --lambda 0.5 0.1 0.001 --epochs=5000
```

`run-all.sh` has all the commands used to run the assignment.


## a1-plot.py

Plot using the data output from `a1.py`.
See `plot-all.sh` for examples.