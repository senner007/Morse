import numpy as np

def min_n_letters(letters):
    def min_n_letters(dataFrame, fileNames):
        mask = (dataFrame['WORD'].str.split(expand=True).apply(lambda x: x.str.len()) >= letters).any(axis=1)
        return dataFrame[mask], fileNames[mask]
    return min_n_letters

def takePercent(percent):
    def takePercent(dataFrame, fileNames):
        fileNames = fileNames[0:int(fileNames.size * (percent/100))]
        dataFrame = dataFrame[0:int(dataFrame.shape[0] * percent/100)]
        return dataFrame, fileNames
    return takePercent

def tempoInterval(params):
    def takePercent(dataFrame, fileNames):
        mask = (dataFrame['Tempo'] >= params[0]) & (dataFrame['Tempo'] <= params[1])
        return dataFrame[mask], fileNames[mask]
    return takePercent