import re
import math
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info/warnings
import numpy as np
import tensorflow as tf
import librosa     #librosa 0.11.0
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# Load TFLite model
#interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized.tflite')
#interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha.tflite')
#interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_int_8_bit.tflite')
#interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/AI-Belha/Results/Custom_split/STFT/best_model-bee_presence_AI_Belha_optimized_float_16_bit.tflite')
#interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/SBCM_4_classes/best_model-bee_presence_SBCM_epoch110_batch128_patience4.h5')
interpreter = tf.lite.Interpreter(model_path='/home/zord/PycharmProjects/SBCM_4_classes/best_model-bee_presence_SBCM_optimized_float_16_bit.tflite')





interpreter.allocate_tensors()

# Get input and output details
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
input_details = interpreter.get_input_details()[0]  # Access the first input tensor
output_details = interpreter.get_output_details()[0]  # Similarly for output if needed


# Function to process input data based on model's expected dtype
def prepare_input_for_model(input_data, input_details):
    dtype = input_details['dtype']
    if dtype == np.uint8:
        scale, zero_point = input_details['quantization']
        if scale == 0:
            # No quantization parameters, assume data is already uint8
            return input_data.astype(np.uint8)
        else:
            # Quantize float input data to uint8
            return np.round(input_data / scale + zero_point).astype(np.uint8)
    elif dtype == np.float32:
        # Model expects float32, ensure data is float32
        return input_data.astype(np.float32)
    elif dtype == np.float16:
        # Model expects float32, ensure data is float32
        #return input_data.astype(np.float32)
        pass
    else:
        raise ValueError(f"Unexpected model input dtype: {dtype}")



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
N_CHUNKS = 32



#---------mean summarization function------#
def mean(s, n_chunks):
    m, f = s.shape
    mod = m % n_chunks
    #print(mod)
    if m % n_chunks != 0:
        s = np.delete(s, np.s_[0:mod] , 0)
    stft_mean = []
    split = np.split(s, n_chunks, axis = 0)
    for i in range(0, n_chunks):
        stft_mean.append(split[i].mean(axis=0))
    stft_mean = np.asarray(stft_mean)
    return stft_mean
#------------------------------------------#


#stft
def stft_extraction_from_array(x, sr, n_chunks):
    # x: numpy array of raw audio waveform
    s = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
    summ_s = mean(s, n_chunks)
    return summ_s


#--------feature extraction tools----------#
#stft
def stft_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath, sr=None)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
    summ_s = mean(s, n_chunks)
    return summ_s




#mfccs - as a baseline
def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath, sr=None)
  mfccs = librosa.feature.mfcc(y=x, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN, window='hann', center=True, dtype=np.complex64, pad_mode='reflect') # originally it was just:  librosa.feature.mfcc(y=x, n_mfcc=32, sr=sr)
  return mfccs


def preprocess(npy_path, sr, n_chunks):
    # Load the `.npy` waveform
    waveform = np.load(npy_path)
    # Run your preprocessing
    processed = stft_extraction_from_array(waveform, sr, n_chunks)
    # Add batch dimension if needed
    #return np.expand_dims(processed, axis=0).astype(np.float32)
    return np.expand_dims(processed, axis=0).astype(np.float16)

def run_inference(prepared_input):
    interpreter.set_tensor(input_details['index'], prepared_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    return output_data


def dequantize_output(output_tensor, output_details):
    dtype = output_details['dtype']
    if dtype == np.uint8:
        scale, zero_point = output_details['quantization']
        return (output_tensor.astype(np.float32) - zero_point) * scale
        """ da indentare a sinistra se usato
        elif dtype == np.float16:
            # Cast float16 output to float32 for processing
            return output_tensor.astype(np.float32)
        """
    else:
        # Already in float32 or other float type
        return output_tensor


# Assuming your model outputs probabilities (like softmax)
def compute_loss(y_true, y_pred):
    # y_true: integer label, y_pred: predicted probabilities
    # Use sparse_categorical_crossentropy
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred).numpy()


# Directory containing folders of `.npy` files
base_dir = '/home/zord/PycharmProjects/SBCM_4_classes/Smart_Bee_Colony_Monitor_16bit_only/1sec'


# Assuming your folders are named by class label
results = {}  # {folder_name: {'total': int, 'misclassified': int}}

# Parameters
sample_rate = SR  # replace with your actual sample rate
n_chunks = N_CHUNKS  # replace with your value



#class 0: queen_absent
#class 1: queen_present_newly_accepted
#class 2: queen_present_original
#class 3: queen_present_rejected

# Define your label mapping
label_mapping = {
    'queen_absent': 0,
    'queen_present_newly_accepted': 1,
    'queen_present_original': 2,
    'queen_present_rejected': 3
}
label_names = list(label_mapping.keys())

# Initialize lists
all_true_labels = []
all_predicted_labels = []

# For per-class accuracy
correct_per_class = {label_idx: 0 for label_idx in label_mapping.values()}
total_per_class = {label_idx: 0 for label_idx in label_mapping.values()}

# Initialize accumulators
total_loss = 0.0
num_samples = 0

true_label = '2' #default: class 'original' --> 2
print("os.listdir(base_dir): ", os.listdir(base_dir))
for folder_name in os.listdir(base_dir):
    print("folder_name: ", folder_name)
    print("label_mapping:", label_mapping)
    if folder_name not in label_mapping:
        continue
    true_label_idx = label_mapping[folder_name]

    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    results[folder_name] = {'total': 0, 'misclassified': 0}

    for filename in os.listdir(folder_path):
        #print("filename: ", filename)
        if filename.endswith('.wav'):
            #print("filename: ", filename)
            wav_path = os.path.join(folder_path, filename)
            #print("npy_path: ", wav_path)
            # Preprocess
            # input_data = preprocess_npy(npy_path, sample_rate, n_chunks)
            #input_data = stft_extraction(wav_path, n_chunks)
            input_data = mfccs_extraction(wav_path)
            input_data = np.expand_dims(input_data, axis=0)  # shape becomes (1, 32, 44)
            input_data = np.expand_dims(input_data, axis=-1)  # shape becomes (1, 32, 44, 1)
            #print("input_data.shape: ", input_data.shape)

            # Prepare input based on model's expected dtype
            input_data_prepared = prepare_input_for_model(input_data, input_details)

            # Run inference
            output = run_inference(input_data_prepared)
            # output is a numpy array, check if quantized
            predicted_probs = dequantize_output(output[0], output_details)
            #predicted_probs = output[0]
            predicted_class = np.argmax(output)
            #print("predicted_class: ", predicted_class)

            # Compute loss for this sample
            loss_value = compute_loss(np.array([true_label_idx]), np.array([predicted_probs]))
            total_loss += loss_value[0]

            # Save true and predicted labels
            all_true_labels.append(true_label_idx)
            all_predicted_labels.append(predicted_class)

            # Update per-class counts
            total_per_class[true_label_idx] += 1
            if predicted_class == true_label_idx:
                correct_per_class[true_label_idx] += 1

            results[folder_name]['total'] += 1
            num_samples += 1
            print(num_samples)
            # Map predicted_class to label if needed
            # For example, if you have a label list:
            # label_list = ['class1', 'class2', ...]
            # predicted_label = label_list[predicted_class]
            # For now, just compare with folder_name
            # You need to map predicted class index to label
            # For simplicity, assume predicted_class matches label index
            # and folder_name is the label string
            if str(predicted_class) != true_label_idx:
                results[folder_name]['misclassified'] += 1
                #print("misclassified: ", results[folder_name]['misclassified'])

# Print the results
for label, stats in results.items():
    total = stats['total']
    misclassified = stats['misclassified']
    print(f"Folder/Label: {label}")
    print(f"  Total samples: {total}")
    print(f"  Misclassifications: {misclassified}")
    print(f"  Accuracy: {(total - misclassified) / total * 100:.2f}%\n")

    print("Classification Report:\n")
    print(classification_report(all_true_labels, all_predicted_labels, target_names=label_names))
    # Optional: print confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    print("Confusion Matrix:\n", cm)


print("\n\n\nNEW METRICS\n")
# Calculate overall metrics
average_loss = total_loss / num_samples
accuracy = np.mean(np.array(all_true_labels) == np.array(all_predicted_labels))
classification_rep = classification_report(all_true_labels, all_predicted_labels, target_names=label_names)
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

# Per-class accuracy
per_class_accuracy = {}
for label_idx in label_mapping.values():
    total = total_per_class[label_idx]
    correct = correct_per_class[label_idx]
    accuracy_cls = correct / total if total > 0 else 0
    per_class_accuracy[label_names[label_idx]] = accuracy_cls

# Print results
print(f"Average Loss: {average_loss:.4f}")
print(f"Overall Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)
print("Per-class accuracy:")
for class_name, acc in per_class_accuracy.items():
    print(f"  {class_name}: {acc:.4f}")
