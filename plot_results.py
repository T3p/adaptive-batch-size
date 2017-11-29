import matplotlib.pyplot as plt
import tables


#[N, alpha, k, J, J^]
# 0    1    2  3  4

filename = 'record.h5'
fp = tables.open_file(filename, mode='r')
data = fp.root.data
n_entries = fp.root.data[:,:].shape[0]
plt.plot(range(1,n_entries+1),fp.root.data[:,3])

plt.show()
