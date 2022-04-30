from tensorflow import keras
import numpy as np
from morse_helpers import convert_image_to_array
from noise_generator.noisegen import NoiseHandling
from audio_process.fft import expand_image_dims, train_img_generate
from Image_Generator_helpers import DataSets, set_paths, global_path, Random_Item

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
            ip = processor["func"](processor["params"])
            train_image_lists, batch_y = ip(train_image_lists, batch_y, self.image_target_size)

        arrays = np.array(train_image_lists) , self.label_func(batch_y)
        return arrays

class Image_Generator_RAW(keras.utils.Sequence) :
    
    def __init__(self, image_amount, set_obj: DataSets, FFT_JUMP, batch_size, image_target_size, image_prepocessors, label_func) :
        self.FFT_JUMP = FFT_JUMP
        self.set_obj = set_obj
        self.image_amout = image_amount
        self.batch_size = batch_size
        self.image_target_size = image_target_size
        self.image_prepocessors = image_prepocessors
        self.label_func = label_func
        self.__set_noise()
    
    def __set_noise(self):
        self.cnoise = NoiseHandling()
        self.cnoise.SetFrequencies(6000,2200,200)

    def __apply_noise(self, signal, signal_to_noise_ratio_db: int):
        signal_noise, some_noise = self.cnoise.addNoise(signal, signal_to_noise_ratio_db)
        return signal_noise
        
    def __len__(self) :
        return (np.ceil(len(self.image_amount) / float(self.batch_size))).astype(np.int32)
    
    
    def __getitem__(self, idx) :

        random_sets = [self.set_obj.get_random() for n in range(self.batch_size)]
        random_signals = [self.set_obj.get_item(random_set) for random_set in random_sets]
        signals_shiftet = [np.insert(signal, 0, np.zeros(0), axis=0) for signal in random_signals] ## prepend with 12840 zeros to align with image pixel 200
        signal_noises = [self.__apply_noise(signal, 30) for signal in signals_shiftet]
        images_noise = [train_img_generate(signal_noise, self.FFT_JUMP) for signal_noise in signal_noises]

        for processor in self.image_prepocessors:
            ip = processor["func"](processor["params"])
            train_image_lists, batch_y = ip(images_noise, batch_y, self.image_target_size)

        arrays = np.array(train_image_lists) , self.label_func(batch_y)
        return arrays
