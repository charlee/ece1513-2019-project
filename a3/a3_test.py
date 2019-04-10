from a3_2 import *


data = np.load('test_data.npy')
data = data.astype(np.float32)
np.random.shuffle(data)

result = run_mog(4, data, tol=1e-8, epochs=5000)
save_and_plot_result(result, '2.2-test')
