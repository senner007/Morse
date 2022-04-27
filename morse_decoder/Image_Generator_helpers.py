
import numpy as np
import random
import pandas as pd
from pandas import read_csv
from io import BufferedReader

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

class Buffers_Class:
    data_buffer = BufferedReader
    index_buffer = BufferedReader
    index_array :  np.ndarray
    def __init__(self, data_buffer, index_buffer, index_array):
        self.data_buffer = data_buffer
        self.index_buffer = index_buffer
        self.index_array = index_array

class Random_Item:
    data_buffer = BufferedReader
    index_buffer = BufferedReader
    index_array :  np.ndarray
    csv_row : pd.DataFrame
    def __init__(self, data_buffer, index_buffer, index_array, csv_row):
        self.data_buffer = data_buffer
        self.index_buffer = index_buffer
        self.index_array = index_array
        self.csv_row = csv_row

class DataSets:
    set_paths_list : "list[Set_Paths]"
    csv_files : "list[pd.DataFrame]" = []
    def __init__(self, sets, global_path):
        self.set_paths_list = sets
        self.global_path = global_path
        self.__cache_csv_files()
        self.buffer_store : list[Buffers_Class]= []
        self.__set_buffers()

    def __cache_csv_files(self):
        for set_path in self.set_paths_list:
            csv: pd.DataFrame = read_csv(self.global_path + set_path.csv_file)
            self.csv_files.append(csv)

    def __set_buffers(self):
        for setx in self.set_paths_list:
            dta_buffer: BufferedReader = open(self.global_path + setx.long16_bin, "rb")
            idx_buffer: BufferedReader = open(self.global_path + setx.long16_index, "rb")
            idx_array = np.fromfile(idx_buffer, dtype=np.float64)

            self.buffer_store.append(Buffers_Class(dta_buffer, idx_buffer, np.reshape(idx_array, (-1,3))))

    def get_item_from_csv(self, set_choice, item_choice):
        csv: pd.DataFrame = self.csv_files[set_choice]
        return csv.iloc[item_choice]

    def get_random(self):
        random_range = np.arange(start=0, stop=len(self.set_paths_list), step=1)
        set_choice: int  = random.choice(random_range)
        random_buffers = self.buffer_store[set_choice]
        item_choice = np.random.randint(0 , random_buffers.index_array.shape[0])
        return Random_Item(random_buffers.data_buffer, random_buffers.index_buffer, random_buffers.index_array[item_choice], self.get_item_from_csv(set_choice, item_choice))

    def close_files(self):
        for buffers in self.buffer_store:
            buffers.data_buffer.close()
            buffers.index_buffer.close()
       

def get_item(random_set: Random_Item):

    siglen, currentpos, scalefac = random_set.index_array
    siglen = np.int32(siglen)
    currentpos = np.int32(currentpos)
    data_buffered =  random_set.data_buffer
    #### numpy_data = np.fromfile(data_buffered, dtype=np.int16)
    byte_step = 2

    data_buffered.seek(currentpos * byte_step)
    ints = np.zeros((siglen,), int)
    for x in range(siglen):
        raw = data_buffered.read(byte_step)
        ints[x] = int.from_bytes(raw, byteorder="little", signed=True)
    
    #### print(ints)
    #### print( numpy_data[currentpos:currentpos + siglen])
    return np.float64(ints)*scalefac