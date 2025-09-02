#dividing samples into folds
import run



#here you can choose the classification method:
random = 1 #
           # 0 - 4-fold cross validation
           # 1 - 70-30 random split
           
#here you can choose the approach of feature extraction:
mode = 0 #
         # 0 - mean-STFT
         # 1 - complex- mean - STFT
         # 2 - mean-CQT
         # 3 - MFCCS
         # 4 - STFT without mean spectrogram (input size 513x44)
         # 5 - CQT without mean spectrogram (input size 513x44)

#some global variables
#class 0: queen_absent
#class 1: queen_present_newly_accepted
#class 2: queen_present_original
#class 3: queen_present_rejected
#class 4: unknown_directory
class_names= ['queen_absent', 'queen_present_newly_accepted', 'queen_present_original', 'queen_present_rejected', 'unknown']
target_names= ['queen_absent', 'queen_present_newly_accepted', 'queen_present_original', 'queen_present_rejected', 'unknown']
#we've got five labels, so:
n_outputs = 5 # originally 2
 #here you can choose value of B:
n_chunks = 32 # originally 27, 16
#some directories
# the following will be used only if random == 0
fold1_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold1"
fold2_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold2"
fold3_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold3"
fold4_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold4"

# the following will be used only if random == 1
queen_absent_dir = "Dataset_AI-Belha/Dataset_wav_format_1_second_samples_augmented/user_obs_status_queen/queen_absent"
queen_present_newly_accepted_dir = "Dataset_AI-Belha/Dataset_wav_format_1_second_samples_augmented/user_obs_status_queen/queen_present_newly_accepted"
queen_present_original_dir = "Dataset_AI-Belha/Dataset_wav_format_1_second_samples_augmented/user_obs_status_queen/queen_present_original"
queen_present_rejected_dir = "Dataset_AI-Belha/Dataset_wav_format_1_second_samples_augmented/user_obs_status_queen/queen_present_rejected"
unknown_dir = "Dataset_AI-Belha/Dataset_wav_format_1_second_samples_augmented/user_obs_status_queen/unknown"


#some cnn performance properties
num_epochs = 110 # originally 60, 50
num_batch_size = 64 # originally 64, 128, 145

#####################################################################################

if random == 0:
    run.four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)
elif random == 1:
    run.random_split(queen_absent_dir, queen_present_newly_accepted_dir, queen_present_original_dir, queen_present_rejected_dir, unknown_dir, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)








