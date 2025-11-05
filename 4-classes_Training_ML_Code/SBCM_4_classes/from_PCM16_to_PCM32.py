import os
import librosa
import soundfile as sf
import numpy as np

def convert_wav_pcm16_to_pcm32(input_folder, output_folder):
    """
    Converts all WAV files in the input folder from PCM16 to PCM32 and saves them in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_pcm32.wav"
            output_path = os.path.join(output_folder, output_filename)

            try:
                # Load audio in original sample rate
                y, sr = librosa.load(input_path, sr=None)

                # Save with PCM_32 subtype (float32 format)
                sf.write(output_path, y, sr, subtype='PCM_32')

                print(f"Converted {filename} to PCM32 and saved as {output_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Usage example:
input_folder = "Dataset_SBCM/4_unknown"   # Replace with your input folder path
output_folder = "Dataset_SBCM/4_unknown_1sec_PCM32" # Replace with your output folder path

convert_wav_pcm16_to_pcm32(input_folder, output_folder)

