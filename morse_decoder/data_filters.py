import numpy as np
from Image_Generator_helpers import  DataSets, set_paths, global_path,Random_Item

def min_letter_mask(dataFrame, letters: int):
    return (dataFrame['WORD'].str.split(expand=True).apply(lambda x: x.str.len()) >= letters).any(axis=1)

def tempo_interval_mask(dataFrame, params):
    return (dataFrame['Tempo'] >= params[0]) & (dataFrame['Tempo'] <= params[1])

def min_n_letters_raw(letters):
    def min_n_letters(dataFrame):
        mask = min_letter_mask(dataFrame, letters)
        return dataFrame[mask]
    return min_n_letters

def tempo_interval_raw(params):
    def tempo_interval_curry(dataFrame):
        mask = tempo_interval_mask(dataFrame, params)
        return dataFrame[mask]
    return tempo_interval_curry

def min_n_letters(letters):
    def min_n_letters(dataFrame, fileNames):
        mask = min_letter_mask(dataFrame, letters)
        return dataFrame[mask], fileNames[mask]
    return min_n_letters

def take_percent(percent):
    def take_percent(dataFrame, fileNames):
        fileNames = fileNames[0:int(fileNames.size * (percent/100))]
        dataFrame = dataFrame[0:int(dataFrame.shape[0] * percent/100)]
        return dataFrame, fileNames
    return take_percent

def tempo_interval(params):
    def tempo_interval_curry(dataFrame, fileNames):
        mask = tempo_interval_mask(dataFrame, params)
        return dataFrame[mask], fileNames[mask]
    return tempo_interval_curry

# de-normalize the regression label position from 0-1 to actual pixel position 
def position_labels_de_normalizer(label, max_position=89, min_position=2, shift_max=60):
    return label * (max_position + shift_max - min_position) + min_position