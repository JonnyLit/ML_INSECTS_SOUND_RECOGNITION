import librosa
import numpy as np

def mfccs_extraction(filepath):
    x, sr = librosa.load(filepath, sr=None)
    mfccs = librosa.feature.mfcc(
        y=x,
        n_mfcc=32,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
        window='hann',
        center=True,
        dtype=np.complex64,
        pad_mode='reflect'
    )
    return mfccs

def amplify_data(data):
    max_value = np.max(np.abs(data))
    if max_value == 0:
        max_value = 1
    amplified = data / max_value
    return amplified

def convert_to_16bit_float_binary(data):
    # Amplify data
    amplified_data = amplify_data(data)
    # Convert to 16-bit float
    float16_array = np.float16(amplified_data)
    # Flatten the array
    float16_flat = float16_array.flatten()
    # Convert to bytes
    binary_values = float16_flat.tobytes()
    # Print each float value
    print("16-bit float values: binary_values")
    i=0
    for val in float16_flat:
        print(f"{i}: {val}")
        i=i+1
    return binary_values

# Example usage:
filepath = '16bit_audios/ESP32_model_1_(Hive1)_25-06-2022_15-04-12_segment_000.wav'
mfccs = mfccs_extraction(filepath)
i = 0
print("32-bit float values: mfccs")
for val in mfccs:
    print(f"{i}: {val}")
    i = i + 1

# Convert MFCCs to binary and print values
binary_values = convert_to_16bit_float_binary(mfccs)

"""
# Save binary data to a file
with open('mfccs_16bit_binary.bin', 'wb') as f:
    f.write(binary_values)
"""