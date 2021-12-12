import numpy as np
import random
from numpy import array, exp
from skimage import data, exposure, img_as_float
import scipy

# slower image shift
def shift_image(img, n):
    return scipy.ndimage.shift(img, [0,n,0], cval=0)

def shift_images_randomly(train_images, params, image_target_size):
    shift = random.randint(params[0], params[1])
    if (shift > 0):
        train_images_padded = [np.pad(img, [(0,0),(shift,0), (0,0)], mode='constant')[:, :image_target_size[1]] for img in train_images]
    else:
        train_images_padded = [np.pad(img, [(0,0),(0,abs(shift)), (0,0)], mode='constant')[:, abs(shift): image_target_size[1] + abs(shift)] for img in train_images]
    
    return train_images_padded, shift

def add_sigmoid(img):
    return 1 / (1 + exp(-img))

def add_normalized_noise(img, mean, std):
    img_noise_orig = img + np.random.normal(mean, std, img.shape)
    # scale values back between 0 and 1
    return exposure.rescale_intensity(img_noise_orig, (0,1))