import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 1000000
N_min = 10000
N_max = 10000
for alpha in [0.1, 0.01, 0.001, 0.0001]:
    for i in range(1):
        for estimator in [1]:
            for bound in [1]:
                if estimator+bound==1:
                    continue
                filename = "results/final/eff_N{}_alpha{}_est{}_bound{}__delta{}_sample{}".format(N_min,alpha,estimator,bound,0.95,i+1)
                filename = filename.replace(".","_")
                filename = filename + ".out"

                subprocess.call("python lqg_fixed.py {} {} {} {} {} {} {}".format(N_min,N_max,0.95,estimator,alpha,filename,max_N), shell=True)

