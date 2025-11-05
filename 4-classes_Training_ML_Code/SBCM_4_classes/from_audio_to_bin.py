import os
import tensorflow as tf
import numpy as np


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


# ---------mean summarization function------#
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


# ------------------------------------------#

# --------feature extraction tools----------#
# stft
def stft_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath, sr=None)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                            center=True, dtype=np.complex64, pad_mode='reflect'))
    # m, t, s = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False,
    # return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    summ_s = mean(s, n_chunks)
    return summ_s
# ------------------------------------------#



def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath, sr=None)
  print(x)
  #mfccs = librosa.feature.mfcc(y=x, n_mfcc=32, sr=sr) # originally it was just:  librosa.feature.mfcc(y=x, n_mfcc=32, sr=sr)
  mfccs = librosa.feature.mfcc(y=x, n_mfcc=32, n_mels=128, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect') # originally it was just:  librosa.feature.mfcc(y=x, n_mfcc=32, sr=sr)
  return mfccs


n_chunks = 32



# Function to classify all audio files in a folder
def print_folder(folder_path):
    print(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                print(folder_path + '/' + file)


#input_folder_list = ["little_dataset_to_get_some_bin"]
#output_folder_path = "bin_data/"


#input_folder_list = ["Dataset_SBCM/mini_dataset_test_ARM_NN/no_queen_present_1sec_PCM32"]
#input_folder_list = ["Dataset_SBCM/mini_dataset_test_ARM_NN/original_queen_1sec_PCM32"]
#input_folder_list = ["Dataset_SBCM/mini_dataset_test_ARM_NN/queen_accepted_1sec_PCM32"]
input_folder_list = ["16bit_audios"]

#output_folder_path = "Dataset_SBCM/mini_dataset_test_ARM_NN_bin/no_queen_present_1sec_PCM32"
#output_folder_path = "Dataset_SBCM/mini_dataset_test_ARM_NN_bin/original_queen_1sec_PCM32"
#output_folder_path = "Dataset_SBCM/mini_dataset_test_ARM_NN_bin/queen_accepted_1sec_PCM32"
#output_folder_path = "Dataset_SBCM/wav_folder_test_original_stft_spectrogram_bin/"
output_folder_path = "16bit_audios_binaries/"





for val in input_folder_list:
    print_folder(val)


import os
import numpy as np
from sklearn.metrics import confusion_matrix

import librosa
# import numpy as np #***********************************
from keras.models import load_model



mode='mfcc'
# mode='stft'

# Function to preprocess audio files
def preprocess_audio(filepath, n_chunks, mode):
    if mode == 'stft':
        input_feature = stft_extraction(filepath, n_chunks)
    elif mode == 'mfcc':
        input_feature = mfccs_extraction(filepath)
    else:
        input_feature = stft_extraction(filepath, n_chunks)
    input_feature = np.expand_dims(input_feature, axis=-1)
    return input_feature

# Function to classify all audio files in a folder
def obtain_npy_from_audio(folder_path):
    # print("__ enter def print_classify_folder(folder_path):")


    # Classify each audio file in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                filename, extension = os.path.splitext(file)
                audio_file_path=folder_path + '/' + file
                processed_audio = preprocess_audio(audio_file_path, n_chunks, 'mfcc')
                output_file_name = output_folder_path + filename + '.bin'
                #np.save(output_file_name, processed_audio)


                # Convert from 32 bit float to float16 before saving
                processed_audio_float16 = processed_audio.astype(np.float16)


                with open(output_file_name, 'wb') as f:
                    processed_audio_float16.tofile(f) # as 16 bit float
                    #processed_audio.tofile(f)  # Write the raw byte data to the binary file as 32 bit float
                print(f"Saved {output_file_name}")


for val in input_folder_list:
    print("==================================================================================")
    print("Checking: ", val)
    obtain_npy_from_audio(val)
