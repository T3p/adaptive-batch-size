import matplotlib.pyplot as plt
import tables
import numpy as np

#[N, alpha, k, J, J^]
# 0    1    2  3  4

filename = 'record.h5'
filepath = 'results/' + filename
fp = tables.open_file(filepath, mode='r')
data = fp.root.data
n_entries = fp.root.data[:,:].shape[0]
#plt.plot(range(1,n_entries+1),data[:,3])

cumul_N = np.cumsum(data[:,0])
print cumul_N
plt.plot(cumul_N,data[:,3])

plt.show()
