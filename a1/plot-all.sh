#!/bin/bash

# 1.3
# python a1-plot.py --path=a1-1.3 --type loss accuracy --phase train --filename a1-1.3/loss~*.csv
# python a1-plot.py --path=a1-1.3 --type loss accuracy --phase valid --filename a1-1.3/loss~*.csv
# python a1-plot.py --path=a1-1.3 --type loss accuracy --phase test --filename a1-1.3/loss~*.csv

# 1.4
# python a1-plot.py --path=a1-1.4 --type loss accuracy --phase train --filename a1-1.4/loss~*.csv
# python a1-plot.py --path=a1-1.4 --type loss accuracy --phase valid --filename a1-1.4/loss~*.csv
# python a1-plot.py --path=a1-1.4 --type loss accuracy --phase test -   -filename a1-1.4/loss~*.csv

# 2.2
# python a1-plot.py --path=a1-2.2 --type loss accuracy --phase train valid test --filename a1-2.2/loss~*.csv

# 2.3
# python a1-plot.py --path=a1-2.3 --type loss accuracy --phase train --filename a1-2.3/loss~log~alpha=0.005_reg=0.0.csv a1-1.3/loss~lr~alpha=0.005_reg=0.csv

# 3.2
# python a1-plot.py --path=a1-3.2 --type loss accuracy --phase train --filename a1-3.2/loss~*.csv
# python a1-plot.py --path=a1-3.2 --type loss accuracy --phase valid --filename a1-3.2/loss~*.csv
# python a1-plot.py --path=a1-3.2 --type loss accuracy --phase test --filename a1-3.2/loss~*.csv

# 3.3
# python a1-plot.py --path=a1-3.3 --type loss accuracy --phase train --filename a1-3.3/loss~*.csv

# 3.4 (1)
# python a1-plot.py --path=a1-3.4/1 --type loss accuracy --phase train --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.95_beta2=0.999_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.99_beta2=0.999_epsilon=1e-08_batchsize=500.csv
# python a1-plot.py --path=a1-3.4/1 --type loss accuracy --phase valid --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.95_beta2=0.999_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.99_beta2=0.999_epsilon=1e-08_batchsize=500.csv
# python a1-plot.py --path=a1-3.4/1 --type loss accuracy --phase test --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.95_beta2=0.999_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.99_beta2=0.999_epsilon=1e-08_batchsize=500.csv

# 3.4 (2)
# python a1-plot.py --path=a1-3.4/2 --type loss accuracy --phase train --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.99_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.9999_epsilon=1e-08_batchsize=500.csv
# python a1-plot.py --path=a1-3.4/2 --type loss accuracy --phase valid --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.99_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.9999_epsilon=1e-08_batchsize=500.csv
# python a1-plot.py --path=a1-3.4/2 --type loss accuracy --phase test --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.99_epsilon=1e-08_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.9999_epsilon=1e-08_batchsize=500.csv

# 3.4 (3)
python a1-plot.py --path=a1-3.4/3 --type loss accuracy --phase train --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=1e-09_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=0.0001_batchsize=500.csv
python a1-plot.py --path=a1-3.4/3 --type loss accuracy --phase valid --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=1e-09_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=0.0001_batchsize=500.csv
python a1-plot.py --path=a1-3.4/3 --type loss accuracy --phase test --filename a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=1e-09_batchsize=500.csv a1-3.4/loss~adam_mse~alpha=0.001_reg=0.0_beta1=0.9_beta2=0.999_epsilon=0.0001_batchsize=500.csv