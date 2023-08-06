from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np

# loading data
sampleRate, data = wavfile.read("/Users/sydneythompson/Downloads/252.wav")

# resample to 16KHz
new_rate = 16000
num_samples = round(len(data) * float(new_rate) / sampleRate)
sample = resample(data, num_samples)

# check that resampled
print(data)
print(sample)

# convert to time
length = sample.shape[0] / new_rate
length2 = data.shape[0] / sampleRate
time = np.linspace(0, length, sample.shape[0])
time2 = np.linspace(0, length2, data.shape[0])

# plot with respect to time
plt.plot(time2, data, label="Audio signal")
plt.plot(time, sample, label=" Processed Audio signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Time segmentation
    # segement signal with overlapping windows 
    # data: numpy array of amplitudes
    # chunk_size: number of samples per chunk
    # jump: step size to move window
    # returns list of list of overlapping samples

def TimeSegmentaiton(data, chunk_size, stride):

    chunks = [data[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]
    return chunks

segmented_data = TimeSegmentaiton(sample, 3, 2)

# Frequency domain analysis
    # segmented_data: list of lists of overlapping samples from inital wav file
def FrequencyAnalysis(segmented_data):

    return

# Synthesis



