import subprocess
import os

N_tot = 30000000
N_min = 100
N_max = 100000

for sample in range(1,6):
    for delta in [0.95]:
        for estimator in [1]:
            for bound in range(1,6):
                filename = "X_results/adabatch_est{}_bound{}_delta{}_sample{}".format(estimator,bound,delta,sample)
                filename = filename.replace(".","_")
                filename = filename + ".out"
                subprocess.call("python lqg_adabatch.py {} {} {} {} {} {} {}".format(N_min,N_max,delta,estimator,bound,filename,N_tot), shell=True)
