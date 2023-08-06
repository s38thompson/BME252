import numpy as np
import csv
from scipy import signal
import matplotlib.pyplot as plt

# loading data
dataFile = 'EMG_Datasets.csv'
with open (dataFile, newline='') as csvfile:
    data = list(csv.reader(csvfile))

initDataArr = np.asarray(data)
dataArr = []

for x in initDataArr:
    y = np.char.split(x)
    y = np.array(y.tolist())

num_rows, num_cols = initDataArr.shape

for x in range(0, num_rows-1):
    dataArr.append(initDataArr[x+1])

num_rows = num_rows-1

relaxed = np.zeros((num_rows, num_cols-1))
contracted = np.zeros((num_rows, num_cols-1))

for x in range(0,num_rows):
    relaxed[x][0] = dataArr[x][0]
    relaxed[x][1] = dataArr[x][1]

    contracted[x][0] = dataArr[x][0]
    contracted[x][1] = dataArr[x][2]

# the two emg datasets
x = np.array(relaxed)
y = np.array(contracted)

# now we have two arrays, relaxed and contracted, each with two columns: column one is the time and column two is the EMG response
# now apply filters to both datasets (arrays)

# creating plot of input
t = np.linspace(0, 1000, 10240, False)
#y = np.sin(0.25*np.pi*2*t) # test signal
sig = x
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('Input')
ax1.axis([0, 200, -1, 1])

# digital bandstop filter
sos = signal.butter(10, 50, 'lp', fs=10240, output='sos') #fs is the sampling frequency (1000 Hz), 10th order, 60 critical freq (Hz)
filtered = signal.sosfilt(sos, sig)
sos = signal.butter(20, 85, 'hp', fs=10240, output='sos')
filtered = signal.sosfilt(sos, sig)

# digital bandpass filter
sos = signal.butter(10, 450, 'lp', fs = 10240, output='sos')
filtered = signal.sosfilt(sos, sig)
sos = signal.butter(20, 15, 'hp', fs = 10240, output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After bandstop and bandpass')
ax2.axis([0, 200, -1, 1])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()