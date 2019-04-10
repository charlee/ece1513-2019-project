from common import plot_result

for k in [5, 10, 15, 20, 30]:
    plot_result('2.2.3', 'kmeans-%s' % str(k), figsize=(4, 3.5))
    plot_result('2.2.3', 'mog-%s' % str(k), figsize=(4, 3.5))
