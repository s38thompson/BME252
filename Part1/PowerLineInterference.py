import pandas as pd
import numpy as np
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

print (n)
print(nyq)