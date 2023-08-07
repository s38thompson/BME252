from scipy.io import wavfile
from scipy.signal import resample, butter, lfilter, sosfilt
import matplotlib.pyplot as plt
import numpy as np
import librosa


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def audio_plot(data, sampleRate):
    T = data.shape[0]/sampleRate
    nsamples = data.shape[0]
    t = np.linspace(0, T, nsamples, endpoint=False)
    plt.plot(t, data, label='OG Data')
    plt.show()

# loading data
sampleRate, data = wavfile.read("Input.wav")

# T = data.shape[0]/sampleRate
# nsamples = data.shape[0]
# t = np.linspace(0, T, nsamples, endpoint=False)
# plt.plot(t, data, label='OG Data')


# Using Librosa
data = data.astype(np.float32)
sample = librosa.resample(data, orig_sr=sampleRate, target_sr=16000)

T2 = sample.shape[0]/16000
nsamples2 = sample.shape[0]
t2 = np.linspace(0, T2, nsamples2, endpoint=False)
# plt.plot(t2, sample, label='Sampled Data')

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
def TimeSegmentaiton(data, sampleRate, chunk_size, stride):

    chunks = [data[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]
    duration = len(data) / sampleRate
    t = np.linspace(0, duration, int(duration*sampleRate), endpoint=False)
    t_chunks = [t[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]
    return chunks, t_chunks

segmented_data, t_chunks = TimeSegmentaiton(sample, 16000, 250, 250)
print(len(segmented_data))
# print(t_chunks)
# bandpass filter parameters
center_freq = list(range(100, 7200, 50))
bandwidth = 50

# Process audio
rms_values = []

print(len(sample))
audio_signal = np.array([])
# print(butter_bandpass_filter(segmented_data[0], 25, 75, 16000, 5))

# temp_signal = np.zeros_like(sample)
# temp_signal += butter_bandpass_filter(sample, 700, 1250, 16000, 6)

# T2 = temp_signal.shape[0]/16000
# nsamples2 = temp_signal.shape[0]
# t2 = np.linspace(0, T2, nsamples2, endpoint=False)
# plt.plot(t2, temp_signal, label='Sampled Data')
# print(temp_signal.shape)
# plt.show()

# create array of butterworth filters centered around different frequencies
# filters = []
# print(segmented_data)
for segment, t in zip(segmented_data, t_chunks):
    chunk_array = np.zeros_like(segment)
    for freq in center_freq:
        min = freq - bandwidth / 2
        max = freq + bandwidth / 2
        b, a = butter(5, [min, max], btype='band', fs=16000)
        filtered_chunk = lfilter(b, a, segment)
        # print(filtered_chunk)
        # print("\n")
        # print(filtered_chunks)
        rms_val = np.sqrt(np.mean(filtered_chunk**2))
        # rms_values.append(rms_val)
        sine = rms_val*np.sin(2*np.pi*freq*t)
        chunk_array += sine
        # print(sine)
    audio_signal = np.concatenate((audio_signal, chunk_array))
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

# print(type(sample))

# audio_synthesized_int16 = (sample * 32767).astype(np.int16)

# print(sample.shape)

audio_plot(audio_signal, 16000)

output_filename = "Output.wav"

wavfile.write(output_filename, 16000, audio_signal.astype(np.int16))



