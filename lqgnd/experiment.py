import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 30000000
N_min = 100
N_max = 100000
for i in range(1):
    for delta in [0.95]:
        for estimator in [1]:
            for bound in [1]:
                if estimator+bound==1:
                    continue
                filename = "results/adabatch_est{}_bound{}__delta{}_sample{}".format(estimator,bound,delta,i+1)
                filename = filename.replace(".","_")
                filename = filename + ".out"

                subprocess.call("python lqgnd_adabatch.py {} {} {} {} {} {} {}".format(N_min,N_max,delta,estimator,bound,filename,max_N), shell=True)

