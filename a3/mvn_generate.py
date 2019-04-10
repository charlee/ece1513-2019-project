import numpy as np
import matplotlib.pyplot as plt

d = 2
n = 500
k = 5

points_per_center = int(n / k)
points = []
for i in range(k):
    mu = np.random.rand(d) * 20 - 10
    var = np.random.rand() * 2
    cov = np.eye(d) * var
    points.append(np.random.multivariate_normal(mu, cov, points_per_center))

data = np.vstack(points) 
np.random.shuffle(data)


np.save('test_data.npy', data)

plt.scatter(data[:,0], data[:,1])
plt.show()