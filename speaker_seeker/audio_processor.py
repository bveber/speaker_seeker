###Standard Python packages###
import time
import math
import os
import multiprocessing
from joblib import Parallel, delayed
###Scientific + Math packages
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.fftpack import dct
import numpy as np

__author__ = 'Brandon'

#Global Initializers
num_cores = multiprocessing.cpu_count()
fps = 10
ffmpeg = os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin', 'ffmpeg.exe')


def get_features(wav_filename, overlap=True, model=False):
    print(wav_filename.split('/')[-1])
    t0 = time.time()
    d = 40
    rate, data = wavfile.read(wav_filename)
    if model and 'Simpsons_' in wav_filename:
        data = data[:int(len(data) / 5)]
    #Convert from stereo to mono
    if len(data.shape) > 1:
        data = data.sum(axis=1) / 2
    frameLength = rate / fps
    index = 0
    samples = []
    b, a = butter(6, [300 / (rate / 2), 4000 / (rate / 2)], btype='band')
    data = lfilter(b, a, data)
    while index < len(data) - frameLength:
        if index % 1000 == 0:
            print('%d frames calculated of %d' % (index, len(data)))
        frame = data[index:index + frameLength]
        if np.abs(frame).sum() > 0:
            frameData = data[index:index + frameLength]
            frameData = frameData / max(frameData)
            #frameData = wiener(data[index:index+frameLength],29)
            frame = frameData * np.hamming(frameLength)
            if len(samples) == 0:
                samples = mfcc_features(frame, d, rate)#samples=fft(frame,d)
            else:
                samples = np.vstack((samples, mfcc_features(frame, d, rate)))#fft(frame,d)))
        else:
            if len(samples) == 0:
                samples = np.zeros(d)#mfcc_features(frame,d,rate)#samples=fft(frame,d)
            else:
                samples = np.vstack((samples, np.zeros(d)))
        if overlap:
            index += int(frameLength / 2)
        else:
            index += frameLength
    print("%-23s%5.2f%-3s" % ('getFeature Runtime is: ', time.time()-t0, '(s)'))
    return samples


def get_features_parallel(wav_filename, overlap=True, model=False):
    print(wav_filename.split('/')[-1])
    t0 = time.time()
    d = 40
    rate, data = wavfile.read(wav_filename)
    if model and 'Simpsons_' in wav_filename:
        data = data[:int(len(data) / 5)]
    #Convert from stereo to mono
    if len(data.shape) > 1:
        data = data.sum(axis=1) / 2
    frameLength = rate / fps
    index = 0
    b, a = butter(6, [300 / (rate / 2), 4000 / (rate / 2)], btype='band')
    data = lfilter(b, a, data)
    frameArray = []
    while index < len(data) - frameLength:
        frameArray.append(data[index:index + frameLength])
        if overlap:
            index += int(frameLength / 2)
        else:
            index += frameLength
    samples = Parallel(n_jobs=num_cores, verbose=1)(delayed(get_features_subroutine)(frame, d, rate, frameLength)
                                         for frame in frameArray)
    print("%-23s%5.2f%-3s" % ('getFeature Runtime is: ', time.time()-t0, '(s)'))
    return np.array(samples)


def get_features_subroutine(frame, d, rate, frameLength):
    if np.abs(frame).sum() > 0:
        frame = frame / max(frame)
        #frameData = wiener(data[index:index+frameLength],29)
        frame = frame * np.hamming(frameLength)
        features = mfcc_features(frame, d, rate)
##        if len(samples)==0: samples = MFCC(frame,d,rate)#samples=fft(frame,d)
##        else: samples=np.vstack((samples,MFCC(frame,d,rate)))#fft(frame,d)))
    else:
        features = np.zeros(d)
##        if len(samples)==0: samples = MFCC(frame,d,rate)#samples=fft(frame,d)
##        else: samples = np.vstack((samples,np.zeros(d)))
    return features


def fft(frame, d):
    fft_indices = np.logspace(np.log10(300), np.log10(4000), d + 1)
    freqs = np.abs(np.fft.rfft(frame))
    features = []
    for i in range(len(fft_indices) - 1):
        freqSum = freqs[fft_indices[i]:fft_indices[i + 1]].sum()
        if freqSum == 0:
            freqSum = .01
        features = np.append(features, np.log10(freqSum))
    return features


def mfcc_features(signal, num_coefficients, sample_rate):
    #num_coefficients = 20 # choose the sive of mfcc array
    minimum_frequency = 300
    maximum_frequency = 4000
    complex_spectrum = np.fft.fft(signal)
    power_spectrum = abs(complex_spectrum) ** 2
    mfb = mel_filter_bank(sample_rate / fps, num_coefficients, minimum_frequency, maximum_frequency)
    filtered_spectrum = np.dot(power_spectrum, mfb)
    log_spectrum = np.log(filtered_spectrum)
    dct_spectrum = dct(log_spectrum, type=2)  # MFCC :)
    for i, elem in enumerate(dct_spectrum):
        if elem > 10 ** 8:
            dct_spectrum[i] = 1e4
        elif elem < -10 ** 8:
            dct_spectrum[i] = -1e4
    return dct_spectrum


def mel_filter_bank(block_size, num_coefficients, minimum_frequency, maximum_frequency):
    num_bands = int(num_coefficients)
    max_mel_frequency = int(frequency_to_mel(maximum_frequency))
    min_mel_frequency = int(frequency_to_mel(minimum_frequency))
    # Create a matrix for triangular filters, one row per filter
    filter_matrix = np.zeros((num_bands, block_size))
    mel_range = np.array(range(num_bands + 2))
    mel_center_filters = mel_range * (max_mel_frequency - min_mel_frequency) / (num_bands + 1) + min_mel_frequency
    # each array index represent the center of each triangular filter
    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(mel_center_filters * aux) - 1) / 22050
    aux = 0.5 + 700 * block_size * aux
    aux = np.floor(aux)  # Round down
    center_index = np.array(aux, int)  # Get int values
    for i in range(num_bands):
        start, center, end = center_index[i:i + 3]
        k1 = np.float32(center - start)
        k2 = np.float32(end - center)
        up = (np.array(range(start, center)) - start) / k1
        down = (end - np.array(range(center, end))) / k2
        filter_matrix[i][start:center] = up
        filter_matrix[i][center:end] = down
    return filter_matrix.transpose()


def frequency_to_mel(frequency):
    return 1127.01048 * math.log(1 + frequency / 700.0)


def mel_to_frequency(mel):
    return 700 * (math.exp(mel / 1127.01048 - 1))
