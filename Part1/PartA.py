import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

def bandstop(input):
    # bandstop filter function
    # input: original dataset
    # output: filtered data
    lowcut = 57 # lowpass cutoff frequency
    highcut = 63 # highpass cutoff frequency
    order = 4 # filter order
    nyq = 0.5 * sampling_rate # nyquist rate
    low = lowcut / nyq # adjusted cutoff
    high = highcut / nyq # adjusted cutoff
    b, a = signal.butter(order, [low, high], btype='bandstop') # filter polynomials
    output = signal.filtfilt(b, a, input) # applying filter
    return output # outputting filtered data

def bandpass(input):
    # bandpass filter function
    # input: original dataset
    # output: filtered data
    lowcut2 = 2.1 # highpass cutoff frequency
    highcut2 = 450 # lowpass cutoff frequency
    order2 = 4 # filter order
    nyq2 = 0.5 * sampling_rate # nyquist rate
    low2 = lowcut2 / nyq2 # adjusted cutoff
    high2 = highcut2 / nyq2 # adjusted cutoff
    d, c = signal.butter(order2, [low2, high2], btype='bandpass') # filter polynomials
    output2 = signal.filtfilt(d, c, input) # applying filter
    return output2 # returning filtered data

def rms(input):
    N = 5 # total time
    integral = np.trapz(np.square(input), None, 0.05)
    rms = np.sqrt((1/N)*integral)
    return rms

# read EMG csv file in
datasets = pd.read_csv('EMG_Datasets.csv')
data = pd.DataFrame(datasets) # convert to DataFrame
num_rows = len(data.index) # get length of csv file
relaxed = data['EMG_Relaxed (mV)'].copy() # copy column of relaxed data into new frame
contracted = data['EMG_Contracted (mV)'].copy() # copy column of contracted data into new frame

# code for testing with sinusoids at two frequencies
f1 = 60  # Hz
f2 = 500  # Hz
sampling_rate = 1000  # samples per second
time = 1  # seconds
t = np.linspace(0, time, time * sampling_rate)
dt = 250  # observation
# y = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

# or using contracted emg dataset
y = contracted
# print(rms(y))
# or using relaxed emg dataset
# y = relaxed

# fast fourier transform to get original signal's frequency components
Y = np.abs(np.fft.fft(y))
f = np.arange(0, len(Y)) * sampling_rate/len(Y)
df = int(600 * len(y) / sampling_rate)

# # plotting signals before filtering
# plt.figure(tight_layout=True)
# plt.subplot(211)
# plt.plot(t[:dt], y[:dt])
# plt.title('Contracted, unfiltered')
# plt.grid()
# plt.xlabel('time(s)')

# plt.subplot(212)
# plt.plot(t[:dt], y[:dt])
# plt.title('Relaxed, unfiltered')
# plt.grid()
# plt.xlabel('time(s)')
# plt.show()

# fft plot, titles, axes, etc.
plt.figure(tight_layout=True)
plt.subplot(211)
plt.plot(f[:df], Y[:df])
plt.title('|Unfiltered|')
plt.grid()
plt.xlabel('f(Hz)')

# applying filters one after the other to input data
filtered = bandpass(y)
filtered = bandstop(filtered)

# fft to get filtered signal's frequency components -- should show 60 Hz and
# <0.1 and >450 Hz filtered out
Y = np.abs(np.fft.fft(filtered))
f = np.arange(0, len(Y)) * sampling_rate/len(Y)
df = int(600 * len(filtered) / sampling_rate)

# fft plot
plt.subplot(212)
plt.plot(f[:df], Y[:df])
plt.title('|Filtered|')
plt.grid()
plt.xlabel('f(Hz)')
plt.show()

# plotting signals after filtering
# plt.figure(tight_layout=True)
# plt.subplot(211)
# plt.plot(t[:dt], filtered[:dt])
# plt.title('Contracted, filtered')
# plt.grid()
# plt.xlabel('time(s)')

# plt.subplot(212)
# plt.plot(t[:dt], filtered[:dt])
# plt.title('Relaxed, filtered')
# plt.grid()
# plt.xlabel('time(s)')
# print(rms(filtered))
# plt.show()
