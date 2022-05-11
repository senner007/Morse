
import numpy as np
import random
import pandas as pd
from pandas import read_csv
from io import BufferedReader
import time
np.set_printoptions(suppress=True)

global_path = "training_data/MorseTrainSet_23"

class Set_Paths:
    long16_bin: str
    long16_index: str
    csv_file : str
    def __init__(self, long16_bin, long16_index, csv_file):
        self.long16_bin = long16_bin
        self.long16_index = long16_index
        self.csv_file = csv_file

set_paths = [
    Set_Paths("/RAW23/Long16_23_010.bin", "/RAW23/L16Index_23_010.bin", "/Words_23_010.csv"),
    Set_Paths("/RAW23/Long16_23_001.bin", "/RAW23/L16Index_23_001.bin", "/Words_23_001.csv"),
    Set_Paths("/RAW23/Long16_23_011.bin", "/RAW23/L16Index_23_011.bin", "/Words_23_011.csv"),
    Set_Paths("/RAW23/Long16_23_002.bin", "/RAW23/L16Index_23_002.bin", "/Words_23_002.csv"),
    Set_Paths("/RAW23/Long16_23_012.bin", "/RAW23/L16Index_23_012.bin", "/Words_23_012.csv"),
    Set_Paths("/RAW23/Long16_23_020.bin", "/RAW23/L16Index_23_020.bin", "/Words_23_020.csv"),
    Set_Paths("/RAW23/Long16_23_021.bin", "/RAW23/L16Index_23_021.bin", "/Words_23_021.csv"),
    Set_Paths("/RAW23/Long16_23_022.bin", "/RAW23/L16Index_23_022.bin", "/Words_23_022.csv"),
    Set_Paths("/RAW23/Long16_23_100.bin", "/RAW23/L16Index_23_100.bin", "/Words_23_100.csv")
]

idx_signal_length = "signal_length"
idx_current_position = "current_position"
idx_scale_factor = "scale_factor"

class Random_Item:
    buffer_path : str
    csv_row : pd.DataFrame
    def __init__(self, buffer_path, csv_row):
        self.buffer_path = buffer_path
        self.csv_row = csv_row

class DataSets:
    set_paths_list : "list[Set_Paths]"
    csv_files : "list[pd.DataFrame]" = []
    def __init__(self, set_paths_list, global_path, masks = ""):
        self.set_paths_list = set_paths_list
        self.global_path = global_path
        self.__cache_dataframes()
        self.__apply_masks(masks)

    def __apply_masks(self, masks):
        for mask in masks:
            for idx, csv in enumerate(self.csv_files):
                self.csv_files[idx] = mask(csv)

    def __cache_dataframes(self):
        for set_path in self.set_paths_list:
            csv: pd.DataFrame = read_csv(self.global_path + set_path.csv_file)
            idx_buffer: BufferedReader = open(self.global_path + set_path.long16_index, "rb")
            idx_array = np.reshape(np.fromfile(idx_buffer, dtype=np.float64), (-1,3))
            idx_buffer.close()
            csv[idx_signal_length] = idx_array[:,0]
            csv[idx_current_position] = idx_array[:,1]
            csv[idx_scale_factor] = idx_array[:,2]
            self.csv_files.append(csv)

    def get_item_from_csv(self, set_choice):
        csv: pd.DataFrame = self.csv_files[set_choice]
        return csv.sample()

    def get_random(self):
        random_range = np.arange(start=0, stop=len(self.set_paths_list), step=1)
        set_choice: int  = random.choice(random_range)
        data_buffer_path = self.global_path + self.set_paths_list[set_choice].long16_bin
        return Random_Item(data_buffer_path, self.get_item_from_csv(set_choice))
    
    def get_item(self, random_set: Random_Item):

        csv_row = random_set.csv_row

        siglen = np.int32(csv_row[idx_signal_length].values[0])
        currentpos = np.int32(csv_row[idx_current_position].values[0])
        data_buffered: BufferedReader = open(random_set.buffer_path, "rb")
        #### numpy_data = np.fromfile(data_buffered, dtype=np.int16)
        byte_step = 2

        data_buffered.seek(int(currentpos * byte_step))
        ints = np.zeros((siglen,), int)
        for x in range(siglen):
            raw = data_buffered.read(byte_step)
            ints[x] = int.from_bytes(raw, byteorder="little", signed=True)

        data_buffered.close()
        
        #### print(ints)
        #### print( numpy_data[currentpos:currentpos + siglen])
        return np.float64(ints) * csv_row[idx_scale_factor].values.astype(np.float64)[0]
