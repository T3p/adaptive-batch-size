import subprocess
import os

N_tot = 30000000
N_min = 100
N_max = 100000

for sample in range(1,6):
    for delta in [0.95, 0.75, 0.5]:
        for estimator in [0]:
            for bound in [0]:
                filename = "X_results/adabatch_est{}_bound{}_delta{}_sample{}".format(estimator,bound,delta,sample)
                filename = filename.replace(".","_")
                filename = filename + ".out"
                subprocess.call("python lqg_adabatch.py {} {} {} {} {} {} {}".format(N_min,N_max,delta,estimator,bound,filename,N_tot), shell=True)

for sample in range(1,6):
    for delta in [0.95, 0.75, 0.5, 0.25, 0.05]:
        for estimator in [1]:
            for bound in [1]:
                filename = "X_results/adabatch_est{}_bound{}_delta{}_sample{}".format(estimator,bound,delta,sample)
                filename = filename.replace(".","_")
                filename = filename + ".out"
                subprocess.call("python lqg_adabatch.py {} {} {} {} {} {} {}".format(N_min,N_max,delta,estimator,bound,filename,N_tot), shell=True)
