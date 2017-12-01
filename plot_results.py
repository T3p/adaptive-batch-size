import matplotlib.pyplot as plt
import tables
import numpy as np

#[N, alpha, k, J, J^]
# 0    1    2  3  4

filename = 'chebyshev_gpomdp_095_1.h5'
filepath = 'results/' + filename
fp = tables.open_file(filepath, mode='r')
data = fp.root.data
n_entries = fp.root.data[:,:].shape[0]
cumul_N = np.cumsum(data[:,0])
plt.plot(cumul_N,data[:,3])

plt.show()
