
import sys
import os

"""
# --- Patch librosa.display directly (Attempt 3) ---
try:
    import librosa.display
except ImportError:
    print("librosa.display not found.  Please ensure librosa is installed.")
    sys.exit(1)

# Get the path to the librosa.display module
librosa_display_path = librosa.display.__file__

# Check if it's a .pyc or .pyo file (compiled) and get the original .py file.
if librosa_display_path.endswith('.pyc') or librosa_display_path.endswith('.pyo'):
    librosa_display_path = librosa_display_path[:-1]

if librosa_display_path is not None: #Only attempt this fix if the file is found.
    try:
        with open(librosa_display_path, 'r') as f:
            lines = f.readlines()

        # Find and replace the import statement.  This may need adjustment if the line is different.
        for i, line in enumerate(lines):
            if "from matplotlib import colormaps as mcm" in line:  #This line is causing the error.
                lines[i] = "from matplotlib import cm as mcm\n"  # Replace with the corrected import
                break  # Stop after the first replacement

        #Write the new content back to the file
        with open(librosa_display_path, 'w') as f:
            f.writelines(lines)
        print("Patched librosa.display to fix colormaps import.")

    except Exception as e:
        print(f"Failed to patch librosa.display: {e}")
        print("Please ensure you have write permissions to the librosa installation directory.")
        sys.exit(1)
else:
    print("Could not locate librosa.display file")
    sys.exit(1)
"""

import re
import math
import numpy as np
from scipy.fft import fft
import matplotlib as mpl  # Import matplotlib itself


# inside your visualization code

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#from matplotlib import colormaps as colormaps
import librosa
import librosa.display
import librosa.feature

print("librosa version:", librosa.__version__)
print("matplotlib version:", matplotlib.__version__)

N_FFT = 1024  # originally 1024
WIN_LEN = 1024
HOP_LEN = int(N_FFT / 2)  # originally 512, N_FFT/2
SR = 22050  # originally 8000
N_MELS = 128  # originally 128
N_MFCC = 32  # originally 13
F_MIN = 0
F_MAX = SR / 2  # originally 4000
TOP_DB = 80  # originally 80
AUDIO_SEC = 1




def mean(s, n_chunks):
    m, f = s.shape
    mod = m % n_chunks
    # print(mod)
    if m % n_chunks != 0:
        s = np.delete(s, np.s_[0:mod], 0)
    stft_mean = []
    split = np.split(s, n_chunks, axis=0)
    for i in range(0, n_chunks):
        stft_mean.append(split[i].mean(axis=0))
    stft_mean = np.asarray(stft_mean)
    return stft_mean



# --------feature extraction tools----------#
# stft
def stft_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath, sr=None)
    s = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
    # m, t, s = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False,
    # return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    summ_s = mean(s, n_chunks)
    return summ_s

#mfccs - as a baseline
def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath, sr=None)
  mfccs = librosa.feature.mfcc(y=x, n_mfcc=N_MFCC, sr=SR) # originally n_mfcc=20
  return mfccs


# ------------approach selection------------------#
def feature_extraction(filepath, n_chunks, mode):
    if mode == 0:
        s = stft_extraction(filepath, n_chunks)
    elif mode == 1:
        s = mfccs_extraction(filepath)
    return s
# ------------------------------------------------#







# 1. (a) Time signal normalization
def normalize_audio(y, sr):
    """
    Normalizes the audio signal for average power, length, and bit depth.
    Handles possible issues with zero input.
    """
    if np.max(np.abs(y)) == 0:  # Handle the case where the audio is all zeros
        return y, sr

    # Normalize for power (RMS)
    rms = librosa.feature.rms(y=y)[0]  # RMS returns an array, take the first element
    y_normalized = y / np.sqrt(np.mean(rms ** 2))

    # Normalize to -1 to 1 range (bit depth)
    y_normalized = y_normalized / np.max(np.abs(y_normalized))

    # No explicit length or bit depth adjustment in Librosa, as loading handles it.
    #  The library automatically handles it.
    return y_normalized, sr


# 2. (b) Power Spectrum
def compute_power_spectrum(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN):  # Updated parameters
    """
    Computes the power spectrum from the audio signal.
    """
    # Calculate number of frames
    num_frames = (AUDIO_SEC * SR - N_FFT) // HOP_LEN + 1
    print(f"Number of frames: {num_frames}")

    # Determine start indices of each frame
    frame_starts = [i * HOP_LEN for i in range(num_frames)]
    print(f"Frame start indices: {frame_starts}")

    # Corresponding time in seconds for frame starts
    frame_times = [start / SR for start in frame_starts]  # sr = 8000
    print(f"Frame start times (seconds): {frame_times}")

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=hop_length)  # STFT
    power_spectrum = np.abs(stft) ** 2  # Magnitude squared (power)
    return power_spectrum, stft


# 3. (c) Mel-frequency scale filterbank output
def compute_mel_spectrogram(power_spectrum, sr, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LEN, fmax=F_MAX):  # Updated parameters and added fmax
    """
    Computes the Mel-frequency spectrogram.  The default is a linear scale until 1 kHz.
    """
    mel_spectrogram = librosa.feature.melspectrogram(S=power_spectrum, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmax=fmax)  # fmax added
    return mel_spectrogram


# 4. (d) Log amplitude of Mel-Spectrogram (in decibels)
def compute_log_mel_spectrogram(mel_spectrogram, top_db=TOP_DB):  # Added top_db parameter
    """
    Computes the log amplitude (in dB) of the Mel spectrogram.
    """
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=top_db)  # dB scale, added top_db
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max, top_db=top_db)  # dB scale, added top_db
    return log_mel_spectrogram


# 5. (e) MFCC (Mel-frequency cepstral coefficients)
def compute_mfccs(log_mel_spectrogram, sr=SR, n_mfcc=N_MFCC):
    """
    Computes the MFCCs.
    """
    mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT)  # Pass the log mel spectrogram directly here
    return mfccs


# 6. (f) Normalized MFCCs (Normalization to mitigate noise)
def normalize_mfccs(mfccs):
    """
    Normalizes the MFCCs to have zero mean and unit variance.
    """
    mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)  # Adding a small number to prevent divide by 0
    return mfccs_normalized










n_chunks = 32

# --- Example Usage (and Visualization) ---
if __name__ == '__main__':

    audio_file = '/home/zord/PycharmProjects/SBCM_4_classes/Dataset_SBCM/original_queen_1sec_PCM32/2022-06-16--00-18-58_2__segment0_segment_016.wav' # 016


    # def stft_extraction(filepath, n_chunks):
    x, sr = librosa.load(audio_file, sr=None)
    print("sr: ")
    print(sr)
    #STFT CLASSIC
    #stft_classic = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
    stft_classic = librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')
    # Calculate total duration based on spectrogram shape
    duration_in_seconds = stft_classic.shape[1] * HOP_LEN / sr
    print("duration_in_seconds: ", duration_in_seconds)
    print("stft_classic.shape: ", stft_classic.shape)

    #STFT WITH MEAN
    #stft_with_mean = mean(stft_classic, n_chunks)
    stft_with_mean = mean(np.abs(stft_classic), n_chunks)





    #MFCC
    x, sr = librosa.load(audio_file, sr=None)
    print("sr: ")
    print(sr)
    mfccs = librosa.feature.mfcc(y=x, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')  # originally n_mfcc=20
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("MFCC FROM AUDIO")
    print(f"mfccs.shape: {mfccs.shape}")
    print(mfccs)
    print("------------------------------------------------------")


    # --- Visualization (Optional, but useful) ---
    plt.figure(figsize=(18, 12))
    #cmap_obj = cm.get_cmap('magma')
    #cmap_obj = colormaps.get_cmap('magma')
    #cmap_obj='magma'
    plt.subplot(3, 3, 1)
    librosa.display.waveshow(x, sr=sr)
    plt.title('ORIGINAL AUDIO')

    # Calculate duration
    duration = len(x) / sr
    print("duration: ", duration)
    plt.subplot(3, 3, 2)
    ax2 = plt.gca()  # get current axes
    #librosa.display.specshow(librosa.amplitude_to_db(stft_classic, ref=np.max), x_axis='time', y_axis='linear', cmap=cmap_obj)
    #STFT_classic_dB=librosa.amplitude_to_db(stft_classic, ref=np.max)
    STFT_classic_dB = librosa.amplitude_to_db(np.abs(stft_classic), ref=np.max)
    times = librosa.frames_to_time(np.arange(STFT_classic_dB.shape[1]), sr=sr, hop_length=HOP_LEN)
    print(STFT_classic_dB.shape)
    print(type(STFT_classic_dB))
    #librosa.display.specshow(STFT_classic_dB, cmap='magma')
    #cmap = plt.get_cmap('viridis')
    img2 = librosa.display.specshow(STFT_classic_dB, x_axis='time', y_axis='linear', cmap='viridis', sr=sr, ax=ax2) #lascia sr=sr qui, altrimenti l'asse temporale non matcha bene col tempo reale
    #plt.imshow(STFT_classic_dB, aspect='auto', cmap='viridis')

    ax2.set_xlim([0, duration])  # Set x-axis to match full duration
    ax2.set_ylim(0, sr / 2)
    print("Y-axis limits:", ax2.get_ylim())
    print("Max value in spectrum:", np.max(np.abs(STFT_classic_dB)))
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
    plt.title('STFT CLASSIC')
    #plt.xlim([0, total_duration])


    plt.subplot(3, 3, 3)
    ax3 = plt.gca()  # get current axes
    #librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_with_mean), ref=np.max), sr=sr, hop_length=512, x_axis='time', y_axis='linear')
    #librosa.display.specshow(stft_with_mean, sr=sr, x_axis='time', y_axis='linear', cmap=cmap_obj)
    stft_with_mean_dB = librosa.amplitude_to_db(stft_with_mean, ref=np.max)
    img3 = librosa.display.specshow(stft_with_mean_dB, x_axis='time', y_axis='linear', cmap='viridis', sr=sr, ax=ax3)
    #plt.imshow(stft_with_mean, aspect='auto', cmap='viridis')
    ax3.set_xlim([0, duration])  # Set x-axis to match full duration
    ax3.set_ylim(0, sr / 2)
    plt.colorbar(img3, ax=ax3, format='%+2.0f dB')
    plt.title('STFT MEAN')


    plt.subplot(3, 3, 4)
    ax4 = plt.gca()  # get current axes
    #librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap=cmap_obj)
    img4 = librosa.display.specshow(mfccs, x_axis='time', y_axis='linear', cmap='viridis', sr=sr, ax=ax4)
    #plt.imshow(mfccs, aspect='auto', cmap='viridis')
    ax4.set_xlim([0, duration])  # Set x-axis to match full duration
    ax4.set_ylim(0, sr / 2)
    plt.colorbar(img4, ax=ax4, format='%+2.0f dB')
    plt.title('MFCCs')

    #plt.tight_layout()
    #plt.show()



    x, sr = librosa.load(audio_file, sr=None)  # Load with the correct sampling rate
    print("sr: ")
    print(sr)
    # (a) Normalize
    #y_normalized, sr = normalize_audio(y, sr)

    """
    # (b) Power Spectrum
    #power_spectrum, stft = compute_power_spectrum(y_normalized, sr)
    power_spectrum, stft = compute_power_spectrum(y, sr)
    print(f"power_spectrum.shape: {power_spectrum.shape}")
    # (c) Mel Spectrogram
    mel_spectrogram = compute_mel_spectrogram(power_spectrum, sr)
    print(f"mel_spectrogram.shape: {mel_spectrogram.shape}")
    # (d) Log Mel Spectrogram
    """








    """
    # Compute STFT
    stft = librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')
    power_spectrum = np.abs(stft) ** 2

    # Compute Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(S=power_spectrum, sr=sr, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LEN, fmax=F_MAX)

    
    #log_mel_spectrogram = compute_log_mel_spectrogram(mel_spectrogram)
    #print(f"log_mel_spectrogram.shape: {log_mel_spectrogram.shape}")
    
    #Convert mel spectrogram to decibels
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # (e) MFCCs

    #mfccs = compute_mfccs(log_mel_spectrogram, sr)
    mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=N_MFCC, sr=SR)
    #mfccs = librosa.feature.mfcc(y=x, n_mfcc=32, sr=sr) # directly using the audio signal x
    """

    # 1. Compute STFT
    stft = librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN, window='hann')

    # 2. Compute power spectrum
    power_spectrum = np.abs(stft) ** 2

    # 3. Create mel filter bank
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmax=F_MAX)

    # 4. Apply mel filter bank to power spectrum
    mel_spectrogram = np.dot(mel_filter_bank, power_spectrum)

    # 5. Convert to dB
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)


    # 6. Compute MFCCs from log mel spectrogram
    mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=N_MFCC, dct_type=2, norm='ortho')


    # Calculate total duration based on spectrogram shape
    duration_in_seconds = log_mel_spectrogram.shape[1] * HOP_LEN / sr
    print("duration_in_seconds: ", duration_in_seconds)
    print("log_mel_spectrogram.shape: ", log_mel_spectrogram.shape)





    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("MFCC FROM MEL SPECTROGRAM:")
    print(f"mfccs.shape: {mfccs.shape}")
    print(mfccs)
    print("------------------------------------------------------")
    # (f) Normalize MFCCs
    #mfccs_normalized = normalize_mfccs(mfccs)
    #print(f"mfccs_normalized.shape: {mfccs_normalized.shape}")
    #print(f"mfccs_normalized.shape[0]:{mfccs_normalized.shape[0]}")
    #print(f"mfccs_normalized.shape[1]:{mfccs_normalized.shape[1]}")


    # --- Visualization (Optional, but useful) ---
    #plt.figure(figsize=(16, 12))

    """
    plt.subplot(3, 3, 5)
    #librosa.display.waveshow(y_normalized, sr=sr)
    librosa.display.waveshow(x, sr=sr)
    plt.title('Normalized Time Signal')
    """

    plt.subplot(3, 3, 5)
    ax5 = plt.gca()  # get current axes
    img5 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max), x_axis='time', y_axis='log', cmap='viridis', sr=sr, ax=ax5)
    ax5.set_xlim([0, duration])  # Set x-axis to match full duration
    ax5.set_ylim(0, sr / 2)
    plt.colorbar(img5, ax=ax5, format='%+2.0f dB')
    plt.title('STFT')

    plt.subplot(3, 3, 6)
    ax6 = plt.gca()  # get current axes
    img6 = librosa.display.specshow(power_spectrum, x_axis='time', y_axis='log', cmap='viridis', sr=sr, ax=ax6)
    ax6.set_xlim([0, duration])  # Set x-axis to match full duration
    ax6.set_ylim(0, sr / 2)
    plt.colorbar(img6, ax=ax6, format='%+2.0f dB')
    plt.title('Power Spectrum')

    plt.subplot(3, 3, 7)
    ax7 = plt.gca()  # get current axes
    img7 = librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', cmap='viridis', sr=sr, ax=ax7)
    ax7.set_xlim([0, duration])  # Set x-axis to match full duration
    ax7.set_ylim(0, sr / 2)
    plt.colorbar(img7, ax=ax7, format='%+2.0f dB')
    plt.title('Mel Spectrogram ')

    plt.subplot(3, 3, 8)
    ax8 = plt.gca()  # get current axes
    #librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=HOP_LEN, x_axis='time', y_axis='mel', cmap='viridis')
    img8 = librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', cmap='viridis', sr=sr, ax=ax8)
    ax8.set_xlim([0, duration])  # Set x-axis to match full duration
    ax8.set_ylim(0, sr / 2)
    plt.colorbar(img8, ax=ax8, format='%+2.0f dB')
    plt.title('Log Mel Spectrogram (dB)')

    plt.subplot(3, 3, 9)
    ax9 = plt.gca()  # get current axes
    img9 = librosa.display.specshow(mfccs, x_axis='time', y_axis='linear', cmap='viridis', sr=sr, ax=ax9)
    ax9.set_xlim([0, duration])  # Set x-axis to match full duration
    ax9.set_ylim(0, sr / 2)
    plt.colorbar(img9, ax=ax9, format='%+2.0f dB')
    plt.title('MFCCs')

    """
    plt.subplot(3, 2, 5)
    librosa.display.specshow(mfccs_normalized, sr=sr, hop_length=HOP_LEN, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar()
    plt.title('Normalized MFCCs')
    """
    plt.tight_layout()
    plt.show()






    # Generate mel filter banks
    mel_filters = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)

    # Plot as an image
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_filters, x_axis='linear', y_axis='linear')
    plt.colorbar(format='%+2.0f')
    plt.title('Mel Filter Bank Matrix')
    plt.xlabel('FFT Bin')
    plt.ylabel('Mel Filter Index')
    plt.show()













    ###############################################################################################àà
                            # MEL FILTER BANKS


    import matplotlib.cm as cm

    # Generate mel filter banks
    mel_filters = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)

    # Compute frequency bins
    fft_bins = np.linspace(0, sr / 2, int(1 + N_FFT // 2))
    # Prepare a colormap
    colors = cm.viridis(np.linspace(0, 1, N_MELS))


    # Plot each mel filter as a triangle
    plt.figure(figsize=(16, 6))
    for i in range(N_MELS):
        filter_coeffs = mel_filters[i]
        # Find the non-zero points
        non_zero_indices = np.where(filter_coeffs > 0)[0]
        if len(non_zero_indices) == 0:
            continue
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1]

        # Get the frequency points
        freq_points = fft_bins[non_zero_indices]
        coeffs = filter_coeffs[non_zero_indices]

        # Plot the triangle
        plt.plot([freq_points[0], freq_points[len(freq_points) // 2], freq_points[-1]],
                 [0, 1, 0], color=colors[i], alpha=0.3)

        # Fill the triangle
        plt.fill([freq_points[0], freq_points[len(freq_points) // 2], freq_points[-1]],
                 [0, 1, 0], color=colors[i], alpha=0.3)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Mel Filter Bank Triangles')

    plt.xlim([0, sr / 2])
    plt.ylim([0, 1.2])





    # Example data
    #x = np.linspace(0, 5000, 1000)
    # y = np.sin(2 * np.pi * x / 5000)

    #plt.plot(x, y)

    # Set major ticks at desired positions
    #tick_positions = [1000, 2000, 3000, 4000, 5000,]
    tick_positions = []
    tick_labels = []
    freq=0
    while freq < sr/2:
        freq+= 500
        tick_positions.append(freq)
        str_freq= str(freq)
        tick_labels.append(str_freq)
    #tick_labels = ['1000Hz', '2000Hz', '3000Hz', '4000Hz', '5000Hz']

    plt.xticks(tick_positions, tick_labels)

    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Amplitude')






    plt.show()








