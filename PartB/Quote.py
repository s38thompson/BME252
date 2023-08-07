from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter, sosfilt
import matplotlib.pyplot as plt
import numpy as np
import librosa


# loading data
sampleRate, data = wavfile.read("/Users/sydneythompson/Downloads/252.wav")

# Using Librosa
data = data.astype(np.float32)
sample = librosa.resample(data, orig_sr=sampleRate, target_sr=16000)

# # resample to 16KHz
# new_rate = 16000
# num_samples = round(len(data) * float(new_rate) / sampleRate)
# sample = resample(data, num_samples)

# # convert to time
# length = sample.shape[0] / new_rate
# length2 = data.shape[0] / sampleRate
# time = np.linspace(0, length, sample.shape[0])
# time2 = np.linspace(0, length2, data.shape[0])

# # plot with respect to time
# plt.plot(time2, data, label="Audio signal")
# plt.plot(time, sample, label=" Processed Audio signal")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()

# Time segmentation
    # segment signal with overlapping windows 
    # data: numpy array of amplitudes
    # chunk_size: number of samples per chunk
    # stride: step size to move to next window
    # returns list of list of overlapping samples
def TimeSegmentaiton(data, chunk_size, stride):

    chunks = [data[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]
    return chunks

segmented_data = TimeSegmentaiton(sample, 3, 3)

# bandpass filter parameters
center_freq = list(range(100, 7200, 500))
bandwidth = 50

# Process audio
rms_values = []
duration = len(sample) / 16000
t = np.linspace(0, duration, int(duration*16000), endpoint=False)
audio_signal = np.zeros_like(sample, dtype=np.float64)
# audio_signal =

# create array of butterworth filters centered around different frequencies
# filters = []

for segment in segmented_data:
    chunk_array = np.zeros_like(segment)
    for freq in center_freq:
        min = freq - bandwidth / 2
        max = freq + bandwidth / 2
        b, a = butter(5, [min, max], btype='band', fs=16000)
        filtered_chunks = lfilter(b, a, segment)
        chunk_array += filtered_chunks
        rms_val = np.sqrt(np.mean(chunk_array**2))
        rms_values.append(rms_val)
        sine = rms_val*np.sin(2*np.pi*freq*t)
        audio_signal += sine
        # filters.append((b, a))



# for chunk in segmented_data:
#     chunk_array = np.zeros_like(chunk)
#     for b, a in filters:
#         filtered_chunk = lfilter(b, a, chunk)
#         chunk_array += filtered_chunk 
#     rms_val = np.sqrt(np.mean(chunk_array**2))
#     rms_values.append(rms_val)
#     for freq in center_freq:
#         sine = np.sin(2*np.pi*freq*t)
    # for rms_val, freq in zip(rms_values, center_freq):
    #     sine = rms_val*np.sin(2*np.pi*freq*t)
    #     audio_signal += sine
    #     # print(rms_val)
    
# print("audio signal: " + audio_signal)

# audio_synthesized_int16 = (sample * 32767).astype(np.int16)

output_filename = "sample_audio.wav"

wavfile.write(output_filename, 16000, audio_signal)



