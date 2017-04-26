import subprocess
import os

#for delta in [float(x)/100 for x in range(100,0,-5)]:
max_N = 30000000
delta = 1 
for i in range(100):
    filename = "results/adabatch_gpomdp_d{}_trial{}".format(delta,i)
    filename = filename.replace(".","_")
    filename = filename + ".out"

    subprocess.call("python lqg_adabatch.py 1000 {} {} {}".format(delta,filename,max_N), shell=True)

