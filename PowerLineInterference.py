import pandas as pd
import numpy as np
import sympy as sy
import math
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


data = pd.read_csv('EMG_Datasets.csv')

# print(data)

# band stop filter- butterworth?

# Filter requirements:
T = 5.0         # Sample Period, seconds 
fs = 2048       # sample rate, Hz : 10240 samples = 2048 samples/second = 1 sample/4.8888x10^-4s : f = 1/t = 1/(5/10240) = 2048
cutoff = 60      # desired cutoff frequency of the filter, Hz , 
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

#PART A.1

#Calculate the power produced by each signal using the root mean square values of the MEG signal (tehcnically the square root of power)
RMS = sqrt(sy.summation(x[n]**2,(n, 0, math.inf))   #Squareing the input ensures they don't all cancel out 
                                                    #Doing the sqaure root will return the result to its original measurement unit (eg. mV (dB)--> mV**2 (dB**2)--> mV (dB))
                                                    #Sine wave's avg is zero however its RMS is approx 0.707*(it's amplitude)


#TODO: Pick an RMS threshold which indicates contracted (above) and relaxed(below)


#PART A.3
#TODO: Remove 60Hz interference using fileter (stop pass? order? cutoff = 60hz?) 

#PART A.4
#TODO: Design bandpass filter to remove frequencies outside 0.1-450 Hz


print (n)
print(nyq)