import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 1000000
N_min = 10000
N_max = 10000

for alpha in [1e-3, 1e-4, 1e-5, 1e-6]:
    for N_min in [10000, 1000, 100]:
        for estimator in [1]:
            for bound in [1]:
                filename = "X_results/fixed_N{}_alpha{}".format(N_min,alpha)
                filename = filename.replace(".","_")
                filename = filename + ".out"

                subprocess.call("python lqg_fixed.py {} {} {} {} {} {} {}".format(N_min,N_max,0.95,estimator,alpha,filename,max_N), shell=True)

