import numpy as np
import random


# Todo : standardize preprocessor to class object and return types.
# Pass in padding bounds parameters 

def add_zeropad_random(params):
    def zeropad_randomly(train_images, lbls, image_target_size):
        n = random.randint(params[0], params[1])
        lbls_np = np.array(lbls) 
        if (n > 0):
            train_images_padded = [np.pad(img, [(0,0),(n,0), (0,0)], mode='constant')[:, :1400] for img in train_images]
        else:
            train_images_padded = [np.pad(img, [(0,0),(0,abs(n)), (0,0)], mode='constant')[:, abs(n): 1400 + abs(n)] for img in train_images]

        lbls_np = lbls_np + (n/image_target_size[1])
        return (train_images_padded, lbls_np)
    
    return zeropad_randomly

def add_noise(params):
    def add_noise(train_images, lbls, image_target_size):
        mean = 0.0   # some constant
        noisy_images = [img + np.random.normal(mean, params, img.shape) for img in train_images]
        return (noisy_images, lbls)
    return add_noise

def add_noise_random(params):
    def add_noise(train_images, lbls, image_target_size):
        r = std = random.randrange(0, 10)
        if r < 5:
            return (train_images, lbls)

        mean = 0.0   # some constant
        std = random.randrange(params[0], params[1])/100
        noisy_images = [img + np.random.normal(mean, std, img.shape) for img in train_images]
        return (noisy_images, lbls)
    
    return add_noise

def cut_and_center(params):
    def cut_and_center(train_images, lbls, image_target_size):
        cut_images = []
        # train_images = train_images.copy()

        for i in range(len(train_images)):
            train_images[i][:, int(lbls[i] * 1400) + 3:] = 0
            padding = int(150 / 2)  - int((int(lbls[i] *1400) + 3) / 2)
            train_image_padded = np.pad(train_images[i], [(0,0),(padding,0), (0,0)], mode='constant')[:, :1400]
            cut_images.append(train_image_padded)

        return (cut_images, lbls)
    return cut_and_center
