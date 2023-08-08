from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np
import librosa

# --------Functions----------- 

# Graphing signal with respect to time
    # data: numpy array containing amplitude values
    # sampleRate: integer representing sampling rate of dataset
    # label: string to label data
    # colour: string to set colour of plot
def audio_plot(data, sampleRate, label, colour):

    # setting parameters
    label = label
    colour = colour

    # converting from frequency to time domain
    T = data.shape[0]/sampleRate
    nsamples = data.shape[0]
    t = np.linspace(0, T, nsamples, endpoint=False)

    # plotting graph
    plt.plot(t, data, label=label, color=colour)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

# Time segmentation; segment signal with overlapping windows 
    # data: numpy array of amplitudes
    # chunk_size: integer of number of samples per chunk
    # stride: integer of step size to move to next window
    # returns list of list of samples and number of samples
def TimeSegmentaiton(data, sampleRate, chunk_size, stride):

    # iterate through array (1D) and break into smaller arrays (2D) containing list of desired chunk length containing amplitude values
    chunks = [data[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]

    # calculate time taken for each sample
    duration = len(data) / sampleRate

    # calculate total time
    t = np.linspace(0, duration, int(duration*sampleRate), endpoint=False)

    # calculate number of chunks and store in list
    t_chunks = [t[i:i+chunk_size] for i in range(0, len(data) - chunk_size + 1, stride)]

    return chunks, t_chunks

# Creating butterworth filter with desired parameters
    # lowcut: integer of min freqeuncy
    # highcut: integer of max frequency
    # fs: integer of sampling rate
    # order: integer of order of butterworth filter
    # returns filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# --------Uploading data----------- 

# loading data
sampleRate, data = wavfile.read("Input.wav")

# plotting origninal wav file
audio_plot(data, sampleRate, "Original", 'b')

# Using Librosa to downsample to 16kHz
data = data.astype(np.float32)
sample = librosa.resample(data, orig_sr=sampleRate, target_sr=16000)

# plotting resampled wav file
audio_plot(sample, 16000, "Resampled", 'g')

# segmenting resampled data into chunks of 250
segmented_data, t_chunks = TimeSegmentaiton(sample, 16000, 250, 250)

# bandpass filter parameters
    # filters with center frequencies from 100-7200, jumping by 50 dB each time
center_freq = list(range(100, 7800, 50))
bandwidth = 50

# --------Processing audio----------- 

# creating dataframes to store rms values and final signal
rms_values = []
audio_signal = np.array([])

# for every chunk and the that time it takes to occur, filter the segment with each filter centered around a different frequency
# calculate the rms value for each specific bandpass-filtered version of the original chunk
# synthesize the sine waves using given equation
# superimpose sine waves to produce final output signal
for segment, t in zip(segmented_data, t_chunks):
    chunk_array = np.zeros_like(segment)
    for freq in center_freq:
        min = freq - bandwidth / 2
        max = freq + bandwidth / 2
        b, a = butter(5, [min, max], btype='band', fs=16000)
        filtered_chunk = lfilter(b, a, segment)
        rms_val = np.sqrt(np.mean(filtered_chunk**2))
        sine = rms_val*np.sin(2*np.pi*freq*t)
        chunk_array += sine
    audio_signal = np.concatenate((audio_signal, chunk_array))

# plotting final output signal
audio_plot(audio_signal, 16000, "Final", 'r')

# setting name of downloaded file
output_filename = "Output.wav"

# downloading final signal with output name
wavfile.write(output_filename, 16000, audio_signal.astype(np.int16))



