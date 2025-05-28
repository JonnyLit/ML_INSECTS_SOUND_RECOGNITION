import tensorflow as tf
import numpy as np
import cv2


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
    x, sr = librosa.load(filepath)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                            center=True, dtype=np.complex64, pad_mode='reflect'))
    # m, t, s = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False,
    # return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    summ_s = mean(s, n_chunks)
    return summ_s
# ------------------------------------------#

# how to call stft_extraction:
# n_chunks = 16
# out = ft.feature_extraction(filepath, n_chunks)

# Replace the import statement for ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rest of your code remains unchanged
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # serve a disattivare la GPU (non viene vista), perciò occorre commentare/decommentare questa riga per attivarla/disattivarla
import numpy as np

import random
import time

import librosa
from scipy import signal

import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras import Model

from keras.utils import to_categorical

# Use keras.layers instead of keras.layers.merge
from keras.layers import Concatenate

from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from skimage.transform import resize
import matplotlib.pyplot as plt


n_chunks = 27



# Function to classify all audio files in a folder
def print_folder(folder_path):
    print(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                print(folder_path + '/' + file)

noqueen = "HoneyBeeQueenPresenceDetectionDataset/TBON/Standardized_Dataset/noqueen_mix"
queen = "HoneyBeeQueenPresenceDetectionDataset/TBON/Standardized_Dataset/queen_0x"

#folder_list = [noqueen, queen]
folder_list = [queen]

for val in folder_list:
    print_folder(val)



target_names = ['noqueen', 'queen'] # label delle classi


import os
import numpy as np
from sklearn.metrics import confusion_matrix

import librosa
# import numpy as np #***********************************
from keras.models import load_model

# Load the saved model
model = TFLiteModel('queen_bee_presence_prediction.tflite')

# Function to preprocess audio files
def preprocess_audio(filepath, n_chunks):
    input_feature = stft_extraction(filepath, n_chunks)
    input_feature = np.expand_dims(input_feature, axis=-1)
    return input_feature

# Function to perform inference on audio files
def classify_audio(audio_path, n_chunks):
    #print("__ enter def classify_audio(audio_path):")
    # Preprocess the audio
    processed_audio = preprocess_audio(audio_path, n_chunks)
    # Perform inference using the loaded model
    prediction = model.predict(np.expand_dims(processed_audio, axis=0))
    # Interpret the prediction
    class_idx = np.argmax(prediction)
    #print("__ exit def classify_audio(audio_path):")
    return class_idx

# Function to classify all audio files in a folder
def classify_folder(folder_path):
    #print("__ enter def classify_folder(folder_path):")
    audio_files = []
    predicted_classes = []
    target_names = ['noqueen', 'queen']  # label delle classi

    # Classify each audio file in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
                predicted_classes.append(classify_audio(os.path.join(root, file), n_chunks))
    #print("__ exit def classify_folder(folder_path):")
    return audio_files, predicted_classes


# Function to classify all audio files in a folder
def print_classify_folder(folder_path):
    # print("__ enter def print_classify_folder(folder_path):")
    target_names = ['noqueen', 'queen']  # label delle classi

    error_count_in_current_folder = 0
    total_files_in_current_folder = 0
    # Classify each audio file in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                total_files_in_current_folder += 1
                audio_file_path=folder_path + '/' + file
                predicted_class_idx = classify_audio(audio_file_path, n_chunks)
                predicted_class_name = target_names[predicted_class_idx]
                if "noqueen" in folder_path:
                    if predicted_class_name != "noqueen":
                        error_count_in_current_folder += 1
                elif "queen" in folder_path:
                    if predicted_class_name != "queen":
                        error_count_in_current_folder += 1
                else:
                    pass
                print(f"The audio file '{audio_file_path}' is classified as '{predicted_class_name}'.") # uncomment this to see the classification output for each file
    return error_count_in_current_folder, total_files_in_current_folder




for val in folder_list:
    print("==================================================================================")
    print("Checking: ", val)
    error_count_in_current_folder, total_files_in_current_folder = print_classify_folder(val)
    print(f"In the folder '{val}' have been counted '{error_count_in_current_folder}'/'{total_files_in_current_folder}' misclassifications")

