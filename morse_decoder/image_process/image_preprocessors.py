import numpy as np
import random
from numpy import array, exp
from skimage import data, exposure, img_as_float
import scipy
from image_process.image_process_helpers import shift_images_randomly, add_normalized_noise

class Preprocess:
  def __init__(self, processor, params):
    self.processor = processor
    self.params = params

def shift_random_update_positions(params):
    def shift_random_update_positions(train_images, labels, image_target_size):
        train_images_padded, n =  shift_images_randomly(train_images, params, image_target_size)
        labels = labels + (n/image_target_size[1])
        return (train_images_padded, labels)
    
    return shift_random_update_positions

def shift_randomly(params):
    def shift_randomly(train_images, labels, image_target_size):
        train_images_padded, n = shift_images_randomly(train_images, params, image_target_size)
        return (train_images_padded, labels)
    
    return shift_randomly

def add_noise(noise_level):
    def add_noise(train_images, labels, image_target_size):
        mean = 0.0
        noisy_images = [add_normalized_noise(img, mean, noise_level) for img in train_images]
        return (noisy_images, labels)
    return add_noise


def add_noise_randomly(params):
    def add_noise_randomly(train_images, labels, image_target_size):
        # skip adding noise every 2nd time 
        r = random.randrange(0, 10)
        if r < 5:
            return (train_images, labels)
        mean = 0.0   # some constant
        std = random.randrange(params[0], params[1])/100
        noisy_images = [add_normalized_noise(img, mean, std) for img in train_images]
        return (noisy_images, labels)
    
    return add_noise_randomly

def cut_and_center(params, cut_margin=3):
    def cut_and_center(train_images, labels, image_target_size):
        cut_images = []
        labels_position = labels[:,0]
        labels_letters = labels[:,1]
        for i in range(len(train_images)):
            train_images[i][:, int(labels_position[i] * image_target_size[1]) + cut_margin:] = 0
            padding = int(params[0] / 2)  - int((int(labels_position[i] *image_target_size[1]) + cut_margin) / 2)
            train_image_padded = np.pad(train_images[i], [(0,0),(padding,0), (0,0)], mode='constant')[:, :image_target_size[1]]
            cut_images.append(train_image_padded)

        return (cut_images, np.array([labels_position, labels_letters]))
    return cut_and_center

def cut_and_right_align(params, cut_margin=5):
    def cut_and_right_align(train_images, labels, image_target_size):
        cut_images = []
        labels_position = labels[:,0]
        labels_letters = labels[:,1]
        for i in range(len(train_images)):
            train_images[i][:, int(labels_position[i] * image_target_size[1]) + cut_margin:] = 0
            pad_width = int(params[0] - (labels_position[i] * image_target_size[1]))
            train_image_padded = np.pad(train_images[i], [(0,0),(pad_width,0), (0,0)], mode='constant')[:, :image_target_size[1]]
            cut_images.append(train_image_padded)

        return (cut_images, np.array([labels_position, labels_letters]))
    return cut_and_right_align