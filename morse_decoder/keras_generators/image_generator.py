from tensorflow import keras
import numpy as np
from morse_helpers import convert_image_to_array

class Image_Generator(keras.utils.Sequence) :
    
    def __init__(self, image_filenames, labels, batch_size, image_target_size, image_prepocessors, label_func) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.image_target_size = image_target_size
        self.image_prepocessors = image_prepocessors
        self.label_func = label_func
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        train_image_lists = []
        for img_name in batch_x:
            img = convert_image_to_array(img_name, self.image_target_size)
            train_image_lists.append(img)

        for processor in self.image_prepocessors:
            ip = processor.processor(processor.params)
            train_image_lists, batch_y = ip(train_image_lists, batch_y, self.image_target_size)

        arrays = np.array(train_image_lists) , self.label_func(batch_y)
        return arrays

