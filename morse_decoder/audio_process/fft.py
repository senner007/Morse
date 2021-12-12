from scipy.fft import fft
import numpy as np

def stfft(signal, jump, normalizer):
    fft_length=128
    signal_length = signal.shape[0]
    wf = np.hamming(fft_length)
    columns = int(np.floor((signal_length-fft_length)/jump))
    intensity_range = int(fft_length/2)
    fft_image = np.zeros((columns, intensity_range), dtype=np.float32)
    
    position = 0
    for k in range(columns):
        signal_slice = signal[position:position+fft_length]
        fft_calc = fft(signal_slice*wf)
        position = position + jump
        fft_image[k] = np.log10(abs(fft_calc[0:intensity_range]+1.e-10))*20
        
    return normalizer(fft_image), columns

def normalizer(fft_image, cut_off=20):
    fft_image_normalized = fft_image-fft_image.max()
    fft_image_normalized = np.clip(fft_image_normalized, -cut_off, 0)
    fft_image_normalized = (fft_image_normalized + cut_off) / cut_off
    fft_image_normalized = fft_image_normalized.T
    return fft_image_normalized

def get_highest_rows(fft_image):
    # Optimize to return correct rows
    return np.argsort(np.sum(fft_image, axis=1))[-2:]

def cut_fft_image(fft_image, highest_rows_indexes):
    return fft_image[highest_rows_indexes.min() -1: highest_rows_indexes.min() +4]

def fit_image_length(fft_image, size=(5,1400)):
    spectrum_ext = np.zeros(size)
    spectrum_ext[:,:fft_image.shape[1]] = fft_image
    return spectrum_ext

def expand_image_dims(fft_image):
    return np.expand_dims(fft_image, axis=0)
