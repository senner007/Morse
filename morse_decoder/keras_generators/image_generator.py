from tensorflow import keras
import numpy as np
from morse_helpers import convert_image_to_array
from noise_generator.noisegen import NoiseHandling
from audio_process.fft import expand_image_dims, train_img_generate
from Image_Generator_helpers import DataSets, set_paths, global_path, Random_Item
from random import randrange


SIGNAL_TO_PIXEL_CONSTANT = 12860


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

        arrays = np.array(train_image_lists), self.label_func(batch_y)
        return arrays

class Image_Generator_RAW(keras.utils.Sequence) :
    
    def __init__(self, 
            image_amount, 
            set_obj: DataSets, 
            FFT_JUMP: int, 
            batch_size: int, 
            image_target_size: int, 
            image_prepocessors, 
            noise_range, 
            random_signal_indent, 
            label_func, 
            label_post_process,
            signal_cut_off: int = 0,
            position_probability: "list[int]" = [0,10] # the probability that a letter at a position will be intented with zeros
        ):
        self.FFT_JUMP = FFT_JUMP
        self.set_obj: DataSets = set_obj
        self.image_amount = image_amount
        self.batch_size = batch_size
        self.image_target_size = image_target_size
        self.image_prepocessors = image_prepocessors
        self.noise_range= noise_range
        self.random_signal_indent = random_signal_indent
        self.label_func = label_func
        self.label_post_process = label_post_process
        self.signal_cut_off = signal_cut_off
        self.signal_cut_off_indent_margin = signal_cut_off + SIGNAL_TO_PIXEL_CONSTANT
        self.position_probability = position_probability

        self.__set_noise()

    
    def __set_noise(self):
        self.cnoise = NoiseHandling()
        self.cnoise.SetFrequencies(6000,2200,200)

    def __apply_noise(self, signal, signal_to_noise_ratio_db: int):
        if (signal.shape[0] < self.signal_cut_off_indent_margin and signal_to_noise_ratio_db < 15):
            signal = np.append(signal, np.zeros((self.signal_cut_off_indent_margin) - signal.shape[0]))
        
        if (signal.shape[0] >= self.signal_cut_off_indent_margin):
            signal = signal[:self.signal_cut_off_indent_margin]
        
        signal_noise, some_noise = self.cnoise.addNoise(signal, signal_to_noise_ratio_db)
        return signal_noise
        
    def __len__(self) :
        return (np.ceil(self.image_amount) / float(self.batch_size)).astype(np.int32)

    def __get_random_position__(self, random_set: Random_Item):
        should_get_position = randrange(self.position_probability[0],self.position_probability[1]) == 1
        if (should_get_position == True):
            p_letter = randrange(len(self.set_obj.positions))
            p_letter_position = random_set.csv_row[self.set_obj.positions[p_letter]].values[0]
            if (p_letter_position != 0):
                return p_letter_position

        return False

    def __insert_zeros_at_position__(self, signal, random_set: Random_Item):
            random_position = self.__get_random_position__(random_set)
            if (random_position != False):
                return np.insert(signal, int((random_position + 2) * (SIGNAL_TO_PIXEL_CONSTANT / 200)), np.zeros(randrange(3500, 10000))) #insert after random letter here
            return signal
    
    def __getitem__(self, idx) :

        random_sets = [self.set_obj.get_random() for n in range(self.batch_size)]
        random_signals = [self.set_obj.get_item(random_set) for random_set in random_sets]

        random_signals_with_spaces = [self.__insert_zeros_at_position__(random_signal, random_sets[idx]) for (idx, random_signal) in enumerate(random_signals)]

        signal_indent = self.random_signal_indent if type(self.random_signal_indent) == int else randrange(*self.random_signal_indent) # get signal indent int if int or else random within range

        signals_shiftet = [np.insert(signal, 0, np.zeros(signal_indent), axis=0) for signal in random_signals_with_spaces] ## prepend with 12840 zeros to align with image pixel 200
        signal_noises = [self.__apply_noise(signal, randrange(self.noise_range[0], self.noise_range[1])) for signal in signals_shiftet]
        images_noise = [train_img_generate(signal_noise, self.FFT_JUMP, self.signal_cut_off_indent_margin) for signal_noise in signal_noises]

        labels = np.array(self.label_func(random_sets))

        for processor in self.image_prepocessors:
            ip = processor["func"](processor["params"])
            images_noise, labels = ip(images_noise, labels, self.image_target_size)

        arrays =  np.array(images_noise), self.label_post_process(labels, self.set_obj, random_sets)
        return arrays
