from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment as am

# loading data
sampleRate, data = wavfile.read("/Users/sydneythompson/Downloads/252.wav")

# resample to 16KHz
new_rate = 16000
num_samples = round(len(data) * float(new_rate) / sampleRate)
# sample = resample(data, data.size//3)
sample = resample(data, num_samples)


# plot wrt time
length = sample.shape[0] / sampleRate
time = np.linspace(0, length, sample.shape[0])
plt.plot(time, sample, label="Audio signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# print(data.shape)
# print(sample.shape)