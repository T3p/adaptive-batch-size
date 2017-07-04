import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 30000000
for i in range(1):
    for delta in [0.95]:
        for estimator in [1]:
            for bound in [5]:
                if estimator+bound==1:
                    continue
                subprocess.call("python adabatch.py", shell=True)

