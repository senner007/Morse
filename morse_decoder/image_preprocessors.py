import numpy as np
import random
from numpy import array, exp
from skimage import data, exposure, img_as_float

# Todo : standardize preprocessor to class object and return types.
# Pass in padding bounds parameters 

def add_zeropad_random(params):
    def zeropad_randomly(train_images, labels, image_target_size):
        n = random.randint(params[0], params[1])
        if (n > 0):
            train_images_padded = [np.pad(img, [(0,0),(n,0), (0,0)], mode='constant')[:, :image_target_size[1]] for img in train_images]
        else:
            train_images_padded = [np.pad(img, [(0,0),(0,abs(n)), (0,0)], mode='constant')[:, abs(n): image_target_size[1] + abs(n)] for img in train_images]

        labels = labels + (n/image_target_size[1])
        return (train_images_padded, labels)
    
    return zeropad_randomly

def add_noise(noise_level):
    def add_noise(train_images, labels, image_target_size):
        mean = 0.0   # some constant
        noisy_images = [normalized_noise(img, mean, noise_level) for img in train_images]
        return (noisy_images, labels)
    return add_noise

def sigmoid(img):
    return 1 / (1 + exp(-img))

def normalized_noise(img, mean, std):
    img_noise_orig = img + np.random.normal(mean, std, img.shape)
    # squish through sigmoid
    img_noise_orig_sig = sigmoid(img_noise_orig)
    # scale values back between 0 and 1
    return exposure.rescale_intensity(img_noise_orig_sig)

def add_noise_random(params):
    def add_noise(train_images, labels, image_target_size):
        r = std = random.randrange(0, 10)
        if r < 5:
            return (train_images, labels)

        mean = 0.0   # some constant
        std = random.randrange(params[0], params[1])/100
        noisy_images = [normalized_noise(img, mean, std) for img in train_images]
        return (noisy_images, labels)
    
    return add_noise

def cut_and_center(params):
    def cut_and_center(train_images, labels, image_target_size):
        cut_images = []
        labels_position = labels[:,0]
        labels_letters = labels[:,1]
        for i in range(len(train_images)):
            train_images[i][:, int(labels_position[i] * image_target_size[1]) + 3:] = 0
            padding = int(params[0] / 2)  - int((int(labels_position[i] *image_target_size[1]) + 3) / 2)
            train_image_padded = np.pad(train_images[i], [(0,0),(padding,0), (0,0)], mode='constant')[:, :image_target_size[1]]
            cut_images.append(train_image_padded)

        return (cut_images, np.array([labels_position, labels_letters]))
    return cut_and_center
