import subprocess
import os

for delta in [float(x)/100 for x in range(100,0,-5)]:
    max_N = 30000000 

    filename = "results/adabatch_gpomdp_d{}_hoeffding".format(delta)
    filename = filename.replace(".","_")
    filename = filename + ".out"

    subprocess.call("python lqg_hoeffding.py 1000 {} {}".format(delta,filename), shell=True)

