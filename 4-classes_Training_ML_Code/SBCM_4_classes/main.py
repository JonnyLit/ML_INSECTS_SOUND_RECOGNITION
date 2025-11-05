#dividing samples into folds
import run



#here you can choose the classification method:
random = 1 #
           # 0 - 4-fold cross validation
           # 1 - 70-30 random split
           
#here you can choose the approach of feature extraction:
mode = 3 #
         # 0 - mean-STFT
         # 1 - complex- mean - STFT
         # 2 - mean-CQT
         # 3 - MFCCS
         # 4 - STFT without mean spectrogram (input size 513x44)activa
         # 5 - CQT without mean spectrogram (input size 513x44)

#some global variables
#class 0: queen_absent
#class 1: queen_present_newly_accepted
#class 2: queen_present_original
#class 3: queen_present_rejected
class_names= ['queen_absent', 'queen_present_newly_accepted', 'queen_present_original', 'queen_present_rejected']
target_names= ['queen_absent', 'queen_present_newly_accepted', 'queen_present_original', 'queen_present_rejected']
#we've got five labels, so:
n_outputs = 4 # originally 2
 #here you can choose value of B:
n_chunks = 32 # originally 27, 16
#some directories
# the following will be used only if random == 0
fold1_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold1"
fold2_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold2"
fold3_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold3"
fold4_directory = "HoneyBeeQueenPresenceDetectionDataset/TBON/Fold4"

# the following will be used only if random == 1
#SBCM_4_classes_32_bits
#queen_absent_dir = "Dataset_SBCM/no_queen_present_1sec_PCM32"
#queen_present_newly_accepted_dir = "Dataset_SBCM/queen_accepted_1sec_PCM32"
#queen_present_original_dir = "Dataset_SBCM/original_queen_1sec_PCM32"
#queen_present_rejected_dir = "Dataset_SBCM/queen_not_accepted_1sec_PCM32"

#SBCM_4_classes_16_bits
queen_absent_dir = "Smart_Bee_Colony_Monitor_16bit_only/queen_absent"
queen_present_newly_accepted_dir = "Smart_Bee_Colony_Monitor_16bit_only/queen_present_newly_accepted"
queen_present_original_dir = "Smart_Bee_Colony_Monitor_16bit_only/queen_present_original"
queen_present_rejected_dir = "Smart_Bee_Colony_Monitor_16bit_only/queen_present_rejected"


#some cnn performance properties
num_epochs = 110 # originally 60, 50, 110
num_batch_size = 105 # originally 64, 128, 145

#####################################################################################

if random == 0:
    run.four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)
elif random == 1:
    run.random_split(queen_absent_dir, queen_present_newly_accepted_dir, queen_present_original_dir, queen_present_rejected_dir, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)








