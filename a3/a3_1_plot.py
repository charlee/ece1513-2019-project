from common import plot_result

plot_result('1.1')

for k in range(1, 6):
    plot_result('1.2', str(k), figsize=(4, 3.5))

for k in range(1, 6):
    plot_result('1.3', str(k), figsize=(4, 3.5))