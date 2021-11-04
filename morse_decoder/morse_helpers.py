from __future__ import print_function

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image as image_process
from tensorflow import keras
import os
import random

def extract_from_bin(filename, image_x, image_y):
    morse_array = np.fromfile(filename, dtype="uint8")

    morse_array_matrix = morse_array.reshape(-1, image_y, image_x)

    return np.array([x.transpose() for x in morse_array_matrix])


def create_morse_images(data_rows_list, foldername):

    morse_images_dir = foldername + "/"

    # Delete all previous images in folder
    for f in os.listdir(morse_images_dir):
        os.remove(os.path.join(morse_images_dir, f))

    fnames = []

    for i in range(len(data_rows_list)):
        fname =  morse_images_dir +"%d" % i + ".png"
        fnames.append(fname)
        cv2.imwrite(fname, data_rows_list[i])

    return np.array(fnames)

def unison_shuffle(arr1, arr2):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]

def round_to_100(n):
    return int(round(n,-2))

def read_label_words(filename):

     # Read morse words
    morse_words = []

    file1 = open(filename, 'r')
    Lines = file1.readlines()[1:]

    for line in Lines:
        morse_words.append(line.rstrip().split(','))

    return np.array(morse_words)



def create_sets(set_names, image_shape, label_func, letter_n):

    extracts = []
    image_h, image_w, channels = image_shape

    for set_name in set_names:
        folder_name = set_name[0]
        file_name = set_name[1]
        ds = extract_from_bin(folder_name + file_name + '.bin', image_h, image_w)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        image_name = create_morse_images(ds, file_name)
        extracts.append(image_name)


    morse_words = np.array([]).reshape(0,22)

    # Todo : use Pandas to get csv here !
    for set_name in set_names:  
        folder_name = set_name[0] 
        file_name = set_name[2]
        r = read_label_words(folder_name + file_name)
        morse_words = np.vstack([morse_words, r])


    return (np.concatenate(extracts), label_func(morse_words[0:,:10], letter_n, image_w))

def convert_image_to_array(image_name, target_size):
    
    img = image_process.load_img(image_name, color_mode="grayscale", target_size=target_size)
    img = image_process.img_to_array(img)
    img = img/255
    return img

class Image_Generator(keras.utils.Sequence) :
    
    def __init__(self, image_filenames, labels, batch_size, image_target_size, image_prepocessors) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.image_target_size = image_target_size
        self.image_prepocessors = image_prepocessors
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        train_image_lists = []
        for img_name in batch_x:
            img = convert_image_to_array(img_name, self.image_target_size)
            train_image_lists.append(img)

        for image_preprocessor in self.image_prepocessors:
            train_image_lists, batch_y = image_preprocessor(train_image_lists, batch_y, self.image_target_size)

        arrays = (np.array(train_image_lists), batch_y)
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
        train_images_padded = [np.pad(img, [(0,0),(n,0), (0,0)], mode='constant')[:, :-1] for img in train_images]
    else:
        train_images_padded = [np.pad(img, [(0,0),(0,abs(n)), (0,0)], mode='constant')[:, abs(n):-1] for img in train_images]

    lbls_np = lbls_np + (n/image_target_size[1])
    return (train_images_padded, lbls_np)