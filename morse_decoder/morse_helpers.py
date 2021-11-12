from __future__ import print_function

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image as image_process
from tensorflow import keras
import os
import random
import decimal

def extract_from_bin(filename, image_x, image_y):
    morse_array = np.fromfile(filename, dtype="uint8")

    morse_array_matrix = morse_array.reshape(-1, image_y, image_x)

    return np.array([x.transpose() for x in morse_array_matrix])


def create_morse_images(data_rows_list, foldername, overwrite):

    morse_images_dir = foldername + "/"

    # # Delete all previous images in folder
    if overwrite:
        for f in os.listdir(morse_images_dir):
            os.remove(os.path.join(morse_images_dir, f))

    fnames = []

    for i in range(len(data_rows_list)):
        fname =  morse_images_dir +"%d" % i + ".png"
        fnames.append(fname)
        if overwrite:
            cv2.imwrite(fname, data_rows_list[i])

    return np.array(fnames)

def unison_shuffle(arr1, arr2):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]

def round_to_100(n):
    return int(round(n,-2))

def read_label_words(csv_file):

     # Read morse words
    morse_words = []

    file1 = open(csv_file, 'r')
    Lines = file1.readlines()[1:]

    for line in Lines:
        morse_words.append(line.rstrip().split(','))

    return np.array(morse_words)



def create_sets(set_names, image_shape, label_funcs, letter_n, overwrite_images):

    total_image_names = []
    image_h, image_w, channels = image_shape
    total_csv_columns = np.array([]).reshape(0,22)

    for set_name in set_names:
        folder_name,file_name, csv_file = set_name
        data_rows = extract_from_bin(folder_name + file_name + '.bin', image_h, image_w)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        image_names = create_morse_images(data_rows, file_name, overwrite_images)
        total_image_names.append(image_names)
        # Todo : use Pandas to get csv here !
        csv_columns = read_label_words(folder_name + csv_file)
        total_csv_columns = np.vstack([total_csv_columns, csv_columns])


    return (np.concatenate(total_image_names), [label_func(total_csv_columns[0:,:10], letter_n, image_w) for label_func in label_funcs])

def convert_image_to_array(image_name, target_size):
    
    img = image_process.load_img(image_name, color_mode="grayscale", target_size=target_size)
    img = image_process.img_to_array(img)
    img = img/255
    return img


class Image_Generator(keras.utils.Sequence) :
    
    def __init__(self, image_filenames, labels, batch_size, image_target_size, image_prepocessors, labels_to_one_hot) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.image_target_size = image_target_size
        self.image_prepocessors = image_prepocessors
        self.labels_to_one_hot = labels_to_one_hot
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_y_positions =  batch_y[:,0]
        batch_y_letters = batch_y[:,1]


        train_image_lists = []
        for img_name in batch_x:
            img = convert_image_to_array(img_name, self.image_target_size)
            train_image_lists.append(img)

        for image_preprocessor in self.image_prepocessors:
            train_image_lists, batch_y_positions = image_preprocessor(train_image_lists, batch_y_positions, self.image_target_size)

        batch_y_letters = self.labels_to_one_hot(batch_y_letters)

        arrays = np.array(train_image_lists) , batch_y_letters
        return arrays


def data_slice(train_data, slice_size):
    slice_size_rounded = round_to_100(len(train_data) * slice_size)
    first_slice, second_slice = train_data[:slice_size_rounded, ...], train_data[slice_size_rounded:, ...]
    return (first_slice, second_slice)


def data_set_create(train_input, labels_input, slice_size):

    train, train_slice = data_slice(train_input, slice_size)

    labels, labels_slice = data_slice(labels_input, slice_size)

    return (train, train_slice, labels, labels_slice)

# Todo : standardize preprocessor to class object and return types.
# Pass in padding bounds parameters 
def zeropad_randomly(train_images, lbls, image_target_size):
    n = random.randint(-10, 15)
    lbls_np = np.array(lbls) 
    if (n > 0):
        train_images_padded = [np.pad(img, [(0,0),(n,0), (0,0)], mode='constant')[:, :1400] for img in train_images]
    else:
        train_images_padded = [np.pad(img, [(0,0),(0,abs(n)), (0,0)], mode='constant')[:, abs(n): 1400 + abs(n)] for img in train_images]

    lbls_np = lbls_np + (n/image_target_size[1])
    return (train_images_padded, lbls_np)

def add_noise_01(train_images, lbls, image_target_size):
    mean = 0.0   # some constant
    std = 0.1    # some constant (standard deviation)
    noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
    return (noisy_images, lbls)

def add_noise_02(train_images, lbls, image_target_size):
    mean = 0.0   # some constant
    std = 0.2    # some constant (standard deviation)
    noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
    return (noisy_images, lbls)

def add_noise_03(train_images, lbls, image_target_size):
    mean = 0.0   # some constant
    std = 0.3    # some constant (standard deviation)
    noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
    return (noisy_images, lbls)

def add_noise_04(train_images, lbls, image_target_size):
    mean = 0.0   # some constant
    std = 0.4    # some constant (standard deviation)
    noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
    return (noisy_images, lbls)

def add_noise_random(train_images, lbls, image_target_size):
    r = std = random.randrange(0, 10)
    if r < 5:
        return (train_images, lbls)

    mean = 0.0   # some constant
    std = random.randrange(0, 20)/100
    noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
    return (noisy_images, lbls)

def create_all_sets(train, labels, TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE, shuffle_before_test_split=True):

    if shuffle_before_test_split == True:
        # shuffle data
        train, labels = unison_shuffle(train, labels)
    
    # get test slice
    train, train_test, labels, labels_test = data_set_create(train, labels, TEST_SPLIT_SIZE)

    if shuffle_before_test_split == False:
        # shuffle data
        train, labels = unison_shuffle(train, labels)

    # get validation slice
    train, train_validation, labels, labels_validation = data_set_create(train, labels, VALIDATION_SPLIT_SIZE)

    return train, labels, train_validation, labels_validation, train_test, labels_test