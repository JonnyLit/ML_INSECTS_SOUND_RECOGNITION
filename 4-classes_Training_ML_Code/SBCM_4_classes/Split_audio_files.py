import os
import librosa
import soundfile as sf
import numpy as np

def split_audio_into_segments(input_folder, output_folder, segment_duration_sec=2):
    """
    Splits audio files into segments of a specified duration, discarding segments shorter than the specified duration.

    Args:
        input_folder (str): The path to the folder containing the audio files.
        output_folder (str): The path to the folder where the segmented audio files will be saved.
        segment_duration_sec (float, optional): The duration of each segment in seconds. Defaults to 2.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)

            try:
                # Load audio
                y, sr = librosa.load(input_path, sr=None)

                segment_samples = int(segment_duration_sec * sr)

                for i in range(0, len(y), segment_samples):
                    segment = y[i:i + segment_samples]

                    # Skip segments shorter than the specified duration
                    if len(segment) < segment_samples:
                        if len(segment) > 0: # If there is something on it, continue to next iteration
                            print(f"Skipping short segment of {filename}")
                            continue
                        else:
                            break # exit for loop

                    # Normalize the audio
                    #segment_normalized = librosa.util.normalize(segment)

                    # Convert to 16-bit audio
                    #segment_int16 = (segment_normalized * 32767).astype(np.int16)

                    # Create output filename
                    output_filename = f"{os.path.splitext(filename)[0]}_segment_{i // segment_samples:03d}.wav"
                    output_path = os.path.join(output_folder, output_filename)

                    # Save the segment
                    #if 5 < i // segment_samples < 8 or 12 < i // segment_samples < 15 or 25 < i // segment_samples < 28 or 35 < i // segment_samples < 38:
                    sf.write(output_path, segment, sr, subtype='PCM_16')
                    #sf.write(output_path, segment, sr, subtype='PCM_32')
                    print(f"Created segment {i // segment_samples + 1} of {filename} and saved to {output_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# calling the function


input_folder = "Smart_Bee_Colony_Monitor_16bit_only/3_queen_accepted"
output_folder = "Smart_Bee_Colony_Monitor_16bit_only/3_queen_accepted_1_sec"

split_audio_into_segments(input_folder, output_folder, segment_duration_sec=1)