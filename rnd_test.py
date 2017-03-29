import numpy as np
import time

I = 10000000

#Generate random numbers in real time
start = time.time()
for i in range(I):
    10+np.random.normal()*2
print 'RT:', time.time() - start

#Presample
start = time.time()
noises = np.random.normal(0,1,I)
for i in range(I):
    10+noises[i]*2
print 'PRE:', time.time() - start
