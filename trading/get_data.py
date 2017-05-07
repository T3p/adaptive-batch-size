#importing necessary libraries
import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt

#Starting a function called anomalies_data
#args: instrument, period in seconds, trading days
def anomalies_data(symbol, period, window):
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period) + '&p=' + str(window)
    url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
    response = urllib2.urlopen(url_root)
    data = response.read().split('\n')
    #actual data starts at index = 7
    #first line contains full timestamp,
    #every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
            #first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period)
                parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), 
                    float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
            except:
                pass # for time zone offsets thrown into data
    dataframe = pd.DataFrame(parsed_data)
    dataframe.columns = ['ts', 'Open', 'High', 'Low', 'Close', 'Volume']
    dataframe.index = dataframe.ts
    del dataframe['ts']
    return dataframe

def interpolate(data,points):
    n = len(data)
    new_close = {}
    new_volume = {}
    close = data['Close']
    volume = data['Volume']
    for i in range(0,n*points):
        k = i/points
        j = i%points
        t = float(j)/points
        key = str(close.keys()[k]) 
        new_key = key[0:-2]
        new_key = "{}{:02}".format(new_key,j)
        new_close[new_key] = close[key]
        new_volume[new_key] = volume[key]
    return pd.DataFrame([new_close,new_volume])
        
         
a =  anomalies_data("SPY",60,15)
b = interpolate(a,60)
print b

