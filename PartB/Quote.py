from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import librosa

# loading data
sampleRate, data = wavfile.read("/Users/sydneythompson/Downloads/252.wav")

# Using Librosa
# data = data.astype(np.float32)
# sample = librosa.resample(data, orig_sr=sampleRate, target_sr=16000)

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
plt.plot(time, sample, label=" Processed Audio signal")
plt.plot(time2, data, label="Audio signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()



