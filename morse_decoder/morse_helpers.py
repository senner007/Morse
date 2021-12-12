from __future__ import print_function
import pandas as pd
import cv2
from tensorflow.keras.preprocessing import image as image_process
import numpy as np
import os

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

def filter_dataset(masks, csv_rows, image_names):
    for mask in masks:
        masks_curry = mask["func"](mask["params"])
        csv_rows, image_names = masks_curry(csv_rows, image_names)
    
    return csv_rows, image_names

def get_data(set_names, image_shape, overwrite_images, masks):

    total_image_names = np.array([])
    image_h, image_w, channels = image_shape
    total_csv_rows = pd.DataFrame()

    for set_name in set_names:
        folder_name,file_name, csv_file = set_name
        data_rows = extract_from_bin(folder_name + file_name + '.bin', image_h, image_w)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        image_names = create_morse_images(data_rows, file_name, overwrite_images)
        csv_rows = pd.read_csv(folder_name + csv_file)

        # Filter data set
        csv_rows_masked, image_names_masked = filter_dataset(masks, csv_rows, image_names)

        total_image_names = np.concatenate([total_image_names, image_names_masked])
        total_csv_rows = total_csv_rows.append(csv_rows_masked)
   
    return total_csv_rows, total_image_names


def create_sets(set_names, image_shape, label_funcs, letter_n, overwrite_images, masks):

    total_csv_rows, total_image_names = get_data(set_names, image_shape, overwrite_images, masks)

    return (total_image_names, [label_func(total_csv_rows, letter_n, image_shape[1]) for label_func in label_funcs])

def convert_image_to_array(image_name, target_size):
    
    img = image_process.load_img(image_name, color_mode="grayscale", target_size=target_size)
    img = image_process.img_to_array(img)
    img = img/255
    return img

def return_label_positions(labels):
    return labels


def data_slice(train_data, slice_size):
    slice_size_rounded = round_to_100(len(train_data) * slice_size)
    first_slice, second_slice = train_data[:slice_size_rounded, ...], train_data[slice_size_rounded:, ...]
    return (first_slice, second_slice)


def data_set_create(train_input, labels_input, slice_size):

    train, train_slice = data_slice(train_input, slice_size)
    labels, labels_slice = data_slice(labels_input, slice_size)

    return (train, train_slice, labels, labels_slice)


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