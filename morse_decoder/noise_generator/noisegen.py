import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft

class NoiseHandling(object): #   def __init__(self, name):
    Fs=6000                # Sample frekvens
    Fhigh=2200              # High Cut-off frequency
    Flow=200                # Low  Cut-off frequency
    WnHigh=2*Fhigh/Fs
    WnLow=2*Flow/Fs
    BfHigh, AfHigh = signal.butter(5,WnHigh, btype='lowpass')
    BfLow, AfLow = signal.butter(4,WnLow, btype='highpass')
  
    def NoiseGen1(self,Tarray):
        s=np.shape(Tarray)
        for k in range(s[1]):
            rx1 = np.random.normal(0,1, s[0])
            rx2=signal.filtfilt(NoiseHandling.BfHigh,NoiseHandling.AfHigh,rx1)
            rx3=signal.filtfilt(NoiseHandling.BfHigh,NoiseHandling.AfHigh,rx2)
            Tarray[:,k]=rx3
        return Tarray

    def addNoise(self,Sarray,SNRdB):
        s=np.shape(Sarray)
        rms1=np.std(Sarray)
        rx1 = np.random.normal(0,1, len(Sarray))
        Anoise=signal.filtfilt(NoiseHandling.BfHigh,NoiseHandling.AfHigh,rx1)
        Anoise=signal.filtfilt(NoiseHandling.BfLow,NoiseHandling.AfLow,Anoise)
        rmsNoise=np.std(Anoise)
     #   Anoise=Anoise
        SNRfac=np.power(10,SNRdB/20.)
        Anoise=Anoise/(SNRfac*rmsNoise)
        Narray=Sarray+Anoise
        return Narray, Anoise
    
    def SetFrequencies(self,Fs,FHigh,Flow):
        NoiseHandling.WnHigh=2*FHigh/Fs
        NoiseHandling.WnLow=2*Flow/Fs
        NoiseHandling.BfHigh, NoiseHandling.AfHigh = signal.butter(5,NoiseHandling.WnHigh, btype='lowpass')
        NoiseHandling.BfLow, NoiseHandling.AfLow = signal.butter(4,NoiseHandling.WnLow, btype='highpass')
        
    def noise_rng(self):
        np.random.default_rng()