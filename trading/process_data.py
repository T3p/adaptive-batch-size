import pandas as pd
import numpy as np

df = pd.read_csv('US1.XOM_160513_170513.txt', sep=" ")
print "read source"
K = 100000

df = df.head(K+int(0.2*K))

df = df[ ~np.isnan(df.Volume)][['Close','Volume']]
# we calculate returns and percentiles, then kill nans
df = df[['Close','Volume']]
df.Volume.replace(0,1,inplace=True) # days shouldn't have zero volume..
df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
df['ClosePctl'] = df.Close.expanding(100).apply(pctrank)
df['VolumePctl'] = df.Volume.expanding(100).apply(pctrank)
df.dropna(axis=0,inplace=True)
R = df.Return
mean_values = df.mean(axis=0)
std_values = df.std(axis=0)
df = (df - np.array(mean_values))/ np.array(std_values)
df['Return'] = R # we don't want our returns scaled
df = df.head(K)
df.to_csv('month0.txt', header=True, index=False, sep=' ', mode='w')
print 'done'
