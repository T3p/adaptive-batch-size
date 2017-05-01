import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 20000000
N_min = 10
N_max = 10000
for i in range(5):
    for delta in [0.95,0.75,0.5,0.25]:
        for estimator in range(2):
            for bound in range(4):
                if estimator+bound==1:
                    continue
                filename = "results/final/adabatch_est{}_bound{}__delta{}_sample{}".format(estimator,bound,delta,i+1)
                filename = filename.replace(".","_")
                filename = filename + ".out"

                subprocess.call("python lqg_adabatch.py {} {} {} {} {} {} {}".format(N_min,N_max,delta,estimator,bound,filename,max_N), shell=True)

