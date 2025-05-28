'''
import librosa
import numpy as np
import os
import soundfile as sf

# --- Augmentation Functions ---

def pitch_shift(samples, sampling_rate, pitch_factor):
    """Shifts the pitch of the audio."""
    return librosa.effects.pitch_shift(samples, sr=sampling_rate, n_steps=pitch_factor)

def time_stretch(samples, rate):
    """Stretches the time (speed) of the audio."""
    return librosa.effects.time_stretch(samples, rate)

def add_noise(samples, noise_factor=0.005):
    """Adds random white noise to the audio."""
    noise = np.random.randn(len(samples))
    augmented_samples = samples + noise_factor * noise
    return augmented_samples

def random_time_shift(samples, sampling_rate, max_shift=0.5):
    """Randomly shifts the audio in time."""
    shift = np.random.randint(int(-max_shift * sampling_rate), int(max_shift * sampling_rate))
    return np.roll(samples, shift)

def change_volume(samples, change_factor):
    """Adjusts the volume of the audio."""
    return samples * change_factor

# Placeholder functions for more advanced techniques - you'll need to implement these

def time_warp(samples):
    """Applies time warping to the audio."""
    # Implementation may require additional libraries or custom logic
    # Example: from scipy.interpolate import interp1d
    # t = np.linspace(0, 1, len(samples))
    # warp = 0.1 * np.sin(2 * np.pi * t)  # Create a warping function
    # t_warped = t + warp
    # interp_func = interp1d(t_warped, samples, kind='linear', fill_value='extrapolate')
    # warped_samples = interp_func(t)
    # return warped_samples
    return samples  # Placeholder

def frequency_mask(samples, freq_mask_param, sr=22050):
    """Applies frequency masking to the audio."""
    # Implementation may require additional libraries or custom logic
    return samples  # Placeholder

def time_mask(samples, time_mask_param):
    """Applies time masking to the audio."""
    # Implementation may require additional libraries or custom logic
    return samples  # Placeholder

def add_reverb(samples):
    """Simulates reverb in the audio."""
    # Implementation may require additional libraries or custom logic
    return samples  # Placeholder

def equalize(samples, gain_db):
    """Applies equalization to the audio."""
    # Implementation may require additional libraries or custom logic
    return samples  # Placeholder

def dynamic_range_compression(samples, threshold=0.1, ratio=4):
    """Applies dynamic range compression to the audio."""
    # Implementation may require additional libraries or custom logic
    return samples  # Placeholder

def augment_audio(samples, sampling_rate):
    """Applies a combination of audio augmentation techniques."""
    augmented_samples = samples.copy()

    # Apply different augmentations with a probability of 0.5
    if np.random.rand() > 0.5:
        augmented_samples = pitch_shift(augmented_samples, sampling_rate, np.random.randint(-4, 4))  # Shift pitch
    if np.random.rand() > 0.5:
        augmented_samples = librosa.effects.time_stretch(augmented_samples, rate=np.random.uniform(0.8, 1.2))  # Stretch time
    if np.random.rand() > 0.5:
        augmented_samples = add_noise(augmented_samples)  # Add noise
    if np.random.rand() > 0.5:
        augmented_samples = random_time_shift(augmented_samples, sampling_rate)  # Random time shift
    if np.random.rand() > 0.5:
        augmented_samples = change_volume(augmented_samples, np.random.uniform(0.5, 1.5))  # Volume adjustment

    # Advanced augmentations (with placeholders)
    if np.random.rand() > 0.5:
        augmented_samples = time_warp(augmented_samples)
    if np.random.rand() > 0.5 and len(augmented_samples) > 500:  # Add a length check
        augmented_samples = time_mask(augmented_samples, time_mask_param=100)
    if np.random.rand() > 0.5 and len(augmented_samples) > 500:  # Add a length check
        augmented_samples = frequency_mask(augmented_samples, freq_mask_param=50, sr=sampling_rate)
    if np.random.rand() > 0.5:
        augmented_samples = add_reverb(augmented_samples)
    if np.random.rand() > 0.5:
        augmented_samples = equalize(augmented_samples, gain_db=6)  # Example gain value

    return augmented_samples

# --- Data Augmentation Script ---

# Specify path for each class
class_dict = {
    "queen_not_present": "/home/zord/PycharmProjects/TF_test/RPW_tf/SmartBeeColonyMonitor_Dataset_16bit_data_AUGMENTED/train/queen_not_present",
    "queen_present_and_newly_accepted": "/home/zord/PycharmProjects/TF_test/RPW_tf/SmartBeeColonyMonitor_Dataset_16bit_data_AUGMENTED/train/queen_present_and_newly_accepted",
    "queen_present_and_rejected": "/home/zord/PycharmProjects/TF_test/RPW_tf/SmartBeeColonyMonitor_Dataset_16bit_data_AUGMENTED/train/queen_present_and_rejected",
    "queen_present_or_original_queen": "/home/zord/PycharmProjects/TF_test/RPW_tf/SmartBeeColonyMonitor_Dataset_16bit_data_AUGMENTED/train/queen_present_or_original_queen"
}

# Target class amount
target_class_count = 3000

for class_name, path in class_dict.items():
    files = os.listdir(path)
    current_count = len(files)

    # Generate augmented samples for this class until we reach the target class count
    for _ in range(target_class_count - current_count):
        random_file = np.random.choice(files)
        file_path = os.path.join(path, random_file)

        # Load audio file
        try:
            samples, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # Augment audio
        augmented_samples = augment_audio(samples, sr)

        # Save the augmented audio file
        output_path = os.path.join(path, f'augmented_{_}_{random_file}')
        try:
            sf.write(output_path, augmented_samples, sr)
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            continue

print("Data augmentation completed!")


'''
















import librosa
import numpy as np
import os
import soundfile as sf


# --- Augmentation Functions ---

def pitch_shift(samples, sampling_rate, pitch_factor):
    """Shifts the pitch of the audio by pitch_factor semitones."""
    return librosa.effects.pitch_shift(samples, sr=sampling_rate, n_steps=pitch_factor)


def add_noise(samples, noise_factor=0.005):
    """Adds random Gaussian noise to the audio."""
    noise = np.random.randn(len(samples))
    return samples + noise_factor * noise


def apply_frequency_mask(mel_spec, freq_mask_param):
    """Applies frequency masking to a Mel-Spectrogram."""
    num_mel_freqs = mel_spec.shape[0]
    freq_mask = np.zeros(num_mel_freqs)
    mask_start = np.random.randint(0, num_mel_freqs - freq_mask_param)
    freq_mask[mask_start:mask_start + freq_mask_param] = 1
    return mel_spec * (1 - freq_mask[:, None])


def apply_time_mask(mel_spec, time_mask_param):
    """Applies time masking to a Mel-Spectrogram."""
    num_time_steps = mel_spec.shape[1]
    time_mask = np.zeros(num_time_steps)
    mask_start = np.random.randint(0, num_time_steps - time_mask_param)
    time_mask[mask_start:mask_start + time_mask_param] = 1
    return mel_spec * (1 - time_mask)


def amplitude_perturbation(samples, perturb_factor=0.1):
    """Applies random amplitude perturbation to the audio."""
    perturbation = np.random.uniform(1 - perturb_factor, 1 + perturb_factor, size=samples.shape)
    return samples * perturbation


def augment_audio(samples, sampling_rate, method):
    """Applies the selected augmentation method to the audio."""
    if method == 'pitch_shift':
        # Applying pitch shifting with predefined steps
        pitch_steps = [-2, -1, 1, 2]
        pitch_factor = np.random.choice(pitch_steps)
        return pitch_shift(samples, sampling_rate, pitch_factor)
    elif method == 'add_noise':
        return add_noise(samples)
    elif method == 'frequency_mask':
        mel_spec = librosa.feature.melspectrogram(samples, sr=sampling_rate)
        freq_mask_param = int(0.05 * mel_spec.shape[0])  # Mask up to 5% of mel channels
        mel_spec = apply_frequency_mask(mel_spec, freq_mask_param)
        return librosa.feature.inverse.mel_to_audio(mel_spec, sr=sampling_rate)
    elif method == 'time_mask':
        mel_spec = librosa.feature.melspectrogram(samples, sr=sampling_rate)
        time_mask_param = int(0.15 * mel_spec.shape[1])  # Mask up to 15% of time steps
        mel_spec = apply_time_mask(mel_spec, time_mask_param)
        return librosa.feature.inverse.mel_to_audio(mel_spec, sr=sampling_rate)
    elif method == 'amplitude_perturbation':
        return amplitude_perturbation(samples)
    else:
        return samples  # Return original if method is unknown


# --- Data Augmentation Script ---

# Specify path for each class
'''
class_dict = {
    "queen_not_present": "/path/to/queen_not_present",
    "queen_present_and_newly_accepted": "/path/to/queen_present_and_newly_accepted",
    "queen_present_and_rejected": "/path/to/queen_present_and_rejected",
    "queen_present_or_original_queen": "/path/to/queen_present_or_original_queen"
}
'''
class_dict = {
    "queen": "HoneyBeeQueenPresenceDetectionDataset/TBON/Standardized_Dataset/queen_augmented_1x"
}

# Augmentation methods to choose from
augmentation_methods = ['pitch_shift', 'add_noise', 'frequency_mask', 'time_mask', 'amplitude_perturbation']

# Target class amount
target_class_count = 15000

for class_name, path in class_dict.items():
    files = os.listdir(path)
    current_count = len(files)

    # Generate augmented samples for this class until we reach target class count
    for _ in range(target_class_count - current_count):
        random_file = np.random.choice(files)
        file_path = os.path.join(path, random_file)

        # Load audio file
        try:
            samples, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        methods_used = ""
        # Randomly select an augmentation method
        method = np.random.choice(augmentation_methods)
        methods_used += method + '-'
        # Augment audio
        augmented_samples = augment_audio(samples, sr, method)
        '''
        # Randomly select an augmentation method
        method = np.random.choice(augmentation_methods)
        methods_used += method + '-'
        # Augment audio
        augmented_samples = augment_audio(augmented_samples, sr, method)
        
        # Randomly select an augmentation method
        method = np.random.choice(augmentation_methods)
        methods_used += method
        # Augment audio
        augmented_samples = augment_audio(augmented_samples, sr, method)
        '''
        # Save the augmented audio file
        output_path = os.path.join(path, f'augmented_{methods_used}_{random_file}')
        try:
            sf.write(output_path, augmented_samples, sr, format='WAV', subtype='FLOAT')
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
            continue

print("Data augmentation completed!")



'''
# Hive_folders Queen, 276'890 files (70.14%):
_1__        SBCM    80'010 files (20.26%):   2022-06-05 : 2022-06-16  2022-06-24  2022-06-25  2022-06-27 : 2022-06-30  2022-07-01 : 2022-07-15
_2__        SBCM    104'970 files (26.59%):  2022-06-05 : 2022-06-16  2022-06-24  2022-06-25  2022-06-27 : 2022-06-30  2022-07-01 : 2022-07-15
GH001       TBON    1'471 files (0.37%):    2022-10-14
CF003       TBON    2'024 files (0.51%):    N.A.
Hive1       NUHIVE    46'276 files (11.72%):   2018-06-12
Hive3       NUHIVE    42'139 files (11.68%):   2017-07-28

# Hive_folders NoQueen, 117'887 files (29.86%):
_1__        SBCM    13'620 files (3.45%):   2022-06-08  2022-06-09  2022-06-25  2022-06-26  2022-06-27
_2__        SBCM    14'220 files (3.60%):   2022-06-08  2022-06-09  2022-06-25  2022-06-26  2022-06-27
CJ001       TBON    574 files (0.14%):      Day:    100 101 102 103 104
CF001       TBON    6 files (0.0015%):        Day:    0   2
Hive1       NUHIVE    44'072 files (11.16%):   2018-05-31
Hive3       NUHIVE    45'395 files (11.50%):   2017-07-12  2017-07-14  2017-07-15

# Hives
_1__        SBCM    96'630  files (24.48%)
_2__        SBCM    116'190 files (29.43%)
GH001       TBON    1'471   files (0.37%)
CF003       TBON    2'024   files (0.51%)
CJ001       TBON    574     files (0.14%)
CF001       TBON    6       files (0.0015%)
Hive1       NUHIVE  90'348  files (22.88%)
Hive3       NUHIVE  87'534  files (22.17%)

# Datasets
SBCM    212'820  files (53.91%)
TBON    4'075   files (1.03%)
NUHIVE  177'882  files (45.06%)
'''

