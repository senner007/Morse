import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tensorflow.keras as keras
from noise_generator.noisegen import NoiseHandling
from morse_label_funcs import code_number
import time


from audio_process.fft import expand_image_dims, train_img_generate
from Image_Generator_helpers import  DataSets, set_paths, global_path
from data_filters import min_n_letters_raw

model_velocity = keras.models.load_model('saved_model_velocity_raw_23')
model_regression = keras.models.load_model('saved_model_regresssion_raw_2')
model_categorical = keras.models.load_model('saved_model_categorical_raw')
model_binary = keras.models.load_model('saved_raw_binary_2')
TRAINING_IMAGE_DIMENSIONS = (5, 1400)
NODGE_IMAGE_PIXEL_AMOUNT = 5 #Push the image to the left to adjust for incorrect position prediction
CATEGORICAL_IMAGE_CROPPED_WIDTH = TRAINING_IMAGE_DIMENSIONS[1] - 1250 #Cropped with during traing of categorical prediction model"
FFT_JUMP = 64
MEAN_TEMPO_OF_TRAINING_DATA = (18 + 25) / 2 # The mean tempo of the training data tempis 


# insert an image into an empty image at start and en position
def insert_image(empty_image, image, start_pos, end_pos):
    empty_image[:, start_pos:end_pos] = image
    return empty_image

def scale_velocity(img_noise):
    velocity_prediction = (model_velocity(expand_image_dims(img_noise))).numpy()[0][0] * 100
    velocity_prediction

    fft_jump_tempo_coefficient = (MEAN_TEMPO_OF_TRAINING_DATA / velocity_prediction)
    rescaled_jump = FFT_JUMP * fft_jump_tempo_coefficient
    return train_img_generate(signal_noise, rescaled_jump)


def left_shift_image(image, pos):
    empty_image = np.zeros([5, 1400])
    empty_image[:, 0: 1400 - pos] = image[:, pos: 1400]
    return empty_image

def show_image(img, width, position = []):
    plt.figure(figsize=(30,5))
    plt.xlim(0, width)
    if len(position) != 0:
        plt.xticks(position)    
    plt.imshow(img)
    plt.show()



Cnoise=NoiseHandling()
Cnoise.SetFrequencies(6000,2200,200)
Cnoise.noise_rng
# From 0 = no noise to -15 = significant noise
# INFO:  table of SNRdb vs. digit in version\n",
# ver \t 0   1   2  3  4\n",
# SNRdB   30  10  5  2  0\n",

def apply_noise(signal, signal_to_noise_ratio_db):
    signal_noise, some_noise = Cnoise.addNoise(signal, signal_to_noise_ratio_db)
    return signal_noise


ORIG_AUDIO_FILE_NAME = 'training_data/MorseTrainSet_23/AUDIO23/BOPAEWITAVZSEE_10400_23_010.wav'
ORIG_AUDIO_FILE_NAME_2 = 'training_data/MorseTrainSet_23/AUDIO23/ZBGLAQEMPZZNBNA_14400_23_021.wav'
ALICE = 'Alice.wav'

SampleRate, signal = wavfile.read(ALICE)
length = signal.shape[0] / SampleRate

signal = np.float32(signal)
max = np.max(signal)
signal = signal / max

signal_to_noise_ratio_db = 30 # From 0 = no noise to -15 = significant noise
signal_noise = apply_noise(signal, signal_to_noise_ratio_db)
img_noise = train_img_generate(signal_noise, FFT_JUMP)

img_noise_rescaled = scale_velocity(img_noise)




print("Correct: ")
# print("BOPAEWITAVZSEE".lower())
# correct = "ZBGLAQEMPZZNBNA".lower()
# print(correct)
alice = "Alice was beginning"
correct = alice.strip().replace(" ", "")
print("----------------------------------")
print("Prediction: ")

letters = []
def letter_sequence(img_noise_rescaled):


    signal_presence = model_binary(expand_image_dims(img_noise_rescaled)).numpy()[0][0]

    if (signal_presence < 0.5):
        print("Signal presence not found")
        print(signal_presence)
        show_image(img_noise_rescaled, 1400,[0, 50, 100])
        return False

    first_letter_position = model_regression(expand_image_dims(img_noise_rescaled)).numpy()[0][0] * 1400 # regression model requires de_normalizing here
    first_letter_position_nodged = first_letter_position + NODGE_IMAGE_PIXEL_AMOUNT


    start_position = int(CATEGORICAL_IMAGE_CROPPED_WIDTH - first_letter_position_nodged)
    image_with_categorical_cropped = insert_image(
        empty_image=np.zeros([5,CATEGORICAL_IMAGE_CROPPED_WIDTH]), 
        image=img_noise_rescaled[:,:int(first_letter_position_nodged)], 
        start_pos=start_position, 
        end_pos= int(first_letter_position_nodged) + start_position
    )

    categorical_prediction = model_categorical(expand_image_dims(image_with_categorical_cropped))
    # print('categorical prediction: ', np.argmax(categorical_prediction))
    
    letters.append(code_number[np.argmax(categorical_prediction)])

    shifted_image = left_shift_image(img_noise_rescaled, int(first_letter_position_nodged))
    letter_sequence(shifted_image)

    # show_image(shifted_image, 1400)

start_time = time.time()
letter_sequence(img_noise_rescaled)
# your code
elapsed_time = time.time() - start_time

prediction_joined = "". join(letters)
print(prediction_joined)
print("Total time: ")
print(elapsed_time)
print("Prediction success: ")
print(prediction_joined.strip() == correct.strip())