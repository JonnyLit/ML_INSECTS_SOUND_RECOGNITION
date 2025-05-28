'''
Introduction

In the following figure one can see a typical example of a recording. This recording was taken in the lab from a part of an infested trunk.
We know that the trunk was probably infested before carrying it in the lab because it had exit tunnels from previous exits of insects.
We have kept the trunk in the lab until insects came out and we confirmed their identity.
Generally, the internal soundscape of a healthy tree –excluding externally induced vibrations - is silent at the level of audio sounds we seek.
If it is infested one expects to hear a train of pulses.
In the second subplot we see the spectrogram (i.e. the frequency composition of the recording as it changes over time).
'''


import math
from parallelplot import parallel_plot
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os

from matplotlib.mlab import window_hanning
from scipy.io import wavfile
import scipy.io


import librosa
import librosa.display
import librosa.feature


FRAME_SIZE = 256  # 512
HOP_SIZE = 128 # 128
split_frequency = 1400 # in Hz

path = 'treevibes/lab/lab/infested/infested_9.wav'
#path = 'debussy.wav'
path2 = 'redhot.wav'
path3 = 'duke.wav'


def plot_signal(x, beta):
    x = x / (.8 * max(x))
    # Show one recording
    plt.subplot(211) # plot che contiene 2 righe e 1 colonna, e l'immagine sarà in posizione 1
    # plt.figure(figsize = (10,8))
    plt.plot(np.linspace(0, len(x) / sr, len(x)), x)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.title('vibration recording')
    plt.grid(True)

    plt.subplot(212) # plot che contiene 2 righe e 1 colonna, e l'immagine sarà in posizione 2
    cmap = plt.get_cmap('viridis')
    vmin = 20 * np.log10(np.max(x)) - 80  # hide anything below -40 dBc
    cmap.set_under(color='k', alpha=None)

    # plt.figure(figsize = (10,8))
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=256, Fs=sr, noverlap=int(256 - 256 / 6), cmap=cmap, vmin=vmin)
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.blackman(FRAME_SIZE), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.bartlett(FRAME_SIZE), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.hamming(FRAME_SIZE), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    # Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.blackman(FRAME_SIZE), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, beta), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.ylabel('frequency [Hz]')
    plt.show()
    return


def plot_signal_subplots(x):
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    x = x / (.8 * max(x))
    # Show one recording
    plt.subplot(231) # griglia di figure, di 2 righe per 3 colonne, con la prima figura di indice 1 (top-left),
    # con indici che aumentano andando da top-left a bottom-right per le prossime figure
    # plt.figure(figsize = (10,8))
    plt.plot(np.linspace(0, len(x) / sr, len(x)), x)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.title('vibration recording')
    plt.grid(True)

    # make up some data in the open interval (0, 1)


    # plot with various axes scales



    cmap = plt.get_cmap('viridis')
    vmin = 20 * np.log10(np.max(x)) - 80  # hide anything below -40 dBc


    # linear
    plt.subplot(232)
    cmap.set_under(color='k', alpha=None)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 0), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    plt.yscale('linear')
    plt.title('Beta=0')
    plt.grid(True)

    # log
    plt.subplot(233)
    cmap.set_under(color='k', alpha=None)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 1), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    plt.yscale('linear')
    plt.title('Beta=1')
    plt.grid(True)

    # symmetric log
    plt.subplot(234)
    cmap.set_under(color='k', alpha=None)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 2), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    plt.yscale('linear')
    plt.title('Beta=2')
    plt.grid(True)

    # logit
    plt.subplot(235)
    cmap.set_under(color='k', alpha=None)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 3), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    plt.yscale('linear')
    plt.title('Beta=3')
    plt.grid(True)


    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

    plt.show()

def plot_parallel_signal(x):
    x = x / (.8 * max(x))
    # Show one recording
    # plt.figure(figsize = (10,8))
    plt.plot(np.linspace(0, len(x) / sr, len(x)), x)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.title('vibration recording')
    plt.grid(True)


    cmap = plt.get_cmap('viridis')
    vmin = 20 * np.log10(np.max(x)) - 80  # hide anything below -40 dBc
    cmap.set_under(color='k', alpha=None)


    P00, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 0), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    print('len(P00): ', len(P00))
    print('len(x): ', len(x))
    print(x[79999])
    P01, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 1), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P02, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 2), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P03, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 3), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P04, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 4), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P10, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 5), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P11, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 6), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P12, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 7), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P13, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 8), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    P14, freqs, bins, im = plt.specgram(x, NFFT=FRAME_SIZE, Fs=sr, window=np.kaiser(FRAME_SIZE, 9), noverlap=HOP_SIZE, cmap=cmap, vmin=vmin)
    X = np.arange(0, len(x)/sr, (len(x)/sr) / len(P00))
    print('len(X): ', len(X))
    figure, axis = plt.subplots(2, 5)


    axis[0, 0].plot(X, P00)
    axis[0, 0].set_title("P00")

    axis[0, 1].plot(X, P01)
    axis[0, 1].set_title("P01")

    axis[0, 2].plot(X, P02)
    axis[0, 2].set_title("P02")

    axis[0, 3].plot(X, P03)
    axis[0, 3].set_title("P03")

    axis[0, 4].plot(X, P04)
    axis[0, 4].set_title("P04")

    axis[1, 0].plot(X, P10)
    axis[1, 0].set_title("P10")

    axis[1, 1].plot(X, P11)
    axis[1, 1].set_title("P11")

    axis[1, 2].plot(X, P12)
    axis[1, 2].set_title("P12")

    axis[1, 3].plot(X, P13)
    axis[1, 3].set_title("P13")

    axis[1, 4].plot(X, P14)
    axis[1, 4].set_title("P14")

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.ylabel('frequency [Hz]')

    plt.show()
    return


def preprocess(data):
    return PCA().fit_transform(data)

def violin(data, fig, axes):
    axes.violinplot(data)


# Gen some fake data
X0 = np.random.uniform(low=-1, high=1, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X1 = np.random.uniform(low=-1, high=1, size=(2, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X2 = np.random.uniform(low=-2, high=2, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X3 = np.random.uniform(low=-3, high=3, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X4 = np.random.uniform(low=-4, high=4, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X5 = np.random.uniform(low=-5, high=5, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X6 = np.random.uniform(low=-6, high=6, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X7 = np.random.uniform(low=-7, high=7, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X8 = np.random.uniform(low=-8, high=8, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y
X9 = np.random.uniform(low=-9, high=9, size=(1, FRAME_SIZE, FRAME_SIZE)) # scegliere quante figure plottare, e le loro dimensioni x,y


parallel_plot(plot_fn=violin, data=X1, grid_shape=(1, 2), preprocessing_fn=None) # scegliere quante righe e colonne, e applicare il preprocessing definito da preprocess(data)
plt.show()





sr, x = wavfile.read(path) # sr = sample rate, x = signal temporal data
# load audio files with librosa
x_librosa, sr_librosa = librosa.load(path)
redhot_librosa, _  = librosa.load(path2)
duke_librosa, _  = librosa.load(path3)
# =====================================================
# Mel filter banks
filter_banks = librosa.filters.mel(n_fft=FRAME_SIZE, sr=sr, n_mels=10)
print("filter_banks.shape", filter_banks.shape)
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks,
                         sr=sr,
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()
# =====================================================




# =====================================================
# Extracting Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=x_librosa, sr=sr_librosa, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=10)
print("mel_spectrogram.shape", mel_spectrogram.shape)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
print("log_mel_spectrogram.shape", log_mel_spectrogram.shape)
plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram,
                         x_axis="time",
                         y_axis="mel",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()
# =====================================================

# =====================================================
# Extracting MFCCs
mfccs = librosa.feature.mfcc(y=x_librosa, n_mfcc=13, sr=sr_librosa)
print("mfccs.shape", mfccs.shape)
# =====================================================

# =====================================================
# Visualising MFCCs
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs,
                         x_axis="time",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()
# =====================================================

# =====================================================
# Computing first/second MFCCs derivatives
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)
print("delta_mfccs.shape", delta_mfccs.shape)
plt.figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs,
                         x_axis="time",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs,
                         x_axis="time",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()

mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
print("mfccs_features.shape", mfccs_features.shape)
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs_features,
                         x_axis="time",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()

deltas_mfccs_features = np.concatenate((delta_mfccs, delta2_mfccs))
print("deltas_mfccs_features.shape", deltas_mfccs_features.shape)
plt.figure(figsize=(25, 10))
librosa.display.specshow(deltas_mfccs_features,
                         x_axis="time",
                         sr=sr_librosa)
plt.colorbar(format="%+2.f")
plt.show()
# =====================================================


#VIDEO 22

# =====================================================
# Extract spectrograms


stft_x = librosa.stft(x_librosa, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
stft_redhot = librosa.stft(redhot_librosa, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
stft_duke = librosa.stft(duke_librosa, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print("stft_x.shape", stft_x.shape)
# =====================================================

# =====================================================
#Calculate Band Energy Ratio
def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""

    frequency_range = sample_rate / 2
    print("frequency_range", frequency_range)
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    print("frequency_delta_per_bin", frequency_delta_per_bin)
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)



def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""

    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, spectrogram.shape[0])

    band_energy_ratio = []

    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)

    return np.array(band_energy_ratio)

split_frequency_bin = calculate_split_frequency_bin(split_frequency, sr_librosa, stft_x.shape[0])
print("split_frequency_bin", split_frequency_bin)
ber_stft_x = band_energy_ratio(stft_x, split_frequency, sr)
ber_stft_redhot = band_energy_ratio(stft_redhot, split_frequency, sr)
ber_stft_duke = band_energy_ratio(stft_duke, split_frequency, sr)
print("len(ber_stft_x)", len(ber_stft_x))
# =====================================================


# =====================================================
# Visualise Band Energy Ratio
frames = range(len(ber_stft_x))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
print("t", t)
print("len(t)", len(t))
plt.figure(figsize=(25, 10))
plt.plot(t, ber_stft_x, color="b")
#plt.plot(t, ber_stft_redhot, color="r")
#plt.plot(t, ber_stft_duke, color="y")
plt.ylim((0, 20000)) # 0 to #Hz
plt.show()
# =====================================================
'''
Window shape for np.kaiser(NFFT, beta)

beta    Window shape
0       Rectangular
5       Similar to a Hamming
6       Similar to a Hanning
8.6     Similar to a Blackman


'''
for iter_beta in range(10):
    plot_signal(x, iter_beta)

#plot_parallel_signal(x)


plot_signal_subplots(x)








'''
Looking at the figure above one may suggest that the detection of borers is an easy task: 
    an envelope follower or a simple thresholding could reveal the impulses.

However, in the field the signals we get, are not that clean for the following reasons:

    a) Depending on the biological cycle of the pest, it can be noisy or cryptic. 
    The pests can be detected during their larva and adult stage when they move and feed, 
    whereas during their egg or pupa state they are silent (although inside the tree). T
    his means that the algorithm should integrate data from longer 
    time spans (daily, weekly, monthly) to infer the infestation state of a tree.

    b) You do not know if there are borers in the tree and even if there are, 
    one cannot know their number and exact location inside the tree. 
    Some of these impulses are feeble because they originate from a location distant to the probe. 
    Depending on the kind of the wood, the probe can detect feeding sounds within a sphere of 1.5-2m radius.

    c) Not all recordings contain impulses even when one surveys an infested tree. 
    It may happen that in the particular time slot the larvae were not active.

    d) Urban spaces are rich in vibrations originating from cars, footsteps and vocalizations of dogs, birds and humans. 
    Some of these vibrations propagate in the wood and reach the metal probe of the device. 
    Therefore, recordings can be very noisy sometimes to the point that 
    external noise dominates over the impulsive sound of the borer (see figure below for an example).

    e) Regarding trees located far from an urban environments, the rain and strong winds that shake the branches 
    and leaves produce external vibrations and impulsive sounds (e.g. raindrops hitting the tree, 
    the shaking of fibrous structures such as palms) that can be picked up by the sensor as well.

In the following figure we see a spectrogram of a recording from an infested tree in the field. 
One can see the characteristic vertical strip in the spectrogram which is indicative of an impulsive audio event. 
This event corresponds to the crack of fibers as the borers feed and move. 
Notice however, the substantial background noise that vibrates the tree.
'''