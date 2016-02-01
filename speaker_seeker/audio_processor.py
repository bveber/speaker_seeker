###Standard Python packages###
import time
import math
import os
import multiprocessing
from joblib import Parallel, delayed
###Scientific + Math packages
from scipy.io import wavfile
from scipy.signal import butter, lfilter, wiener, gaussian, hann
from scipy.fftpack import dct
from scipy.ndimage import filters
import numpy as np
import pandas as pd
import pysrt

__author__ = 'Brandon'

#Global Initializers
num_cores = multiprocessing.cpu_count()
fps = 25
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
    frame_length = np.floor(rate / fps)
    index = 0
    samples = []
    b, a = butter(6, [300 / (rate / 2), 4000 / (rate / 2)], btype='band')
    data = lfilter(b, a, data)
    while index < len(data) - frame_length:
        if index % 1000 == 0:
            print('%d frames calculated of %d' % (index, len(data)))
        frame = data[index:index + frame_length]
        if np.abs(frame).sum() > 0:
            frameData = data[index:index + frame_length]
            frameData = frameData / max(frameData)
            #frameData = wiener(data[index:index+frame_length],29)
            frame = frameData * np.hamming(frame_length)
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
            index += np.floor(frame_length / 2)
        else:
            index += frame_length
    print("%-23s%5.2f%-3s" % ('getFeature Runtime is: ', time.time()-t0, '(s)'))
    return samples


def get_features_parallel(wav_filename, overlap=True, model=False):
    print(wav_filename.split('/')[-1])
    t0 = time.time()
    d = 26
    rate, data = wavfile.read(wav_filename)
    if model and 'Simpsons_' in wav_filename:
        # np.random.seed(42)
        # permutation = np.random.permutation(data.shape[0])
        # data = data[permutation]
        data = data[:len(data) / 10]
    #Convert from stereo to mono
    if len(data.shape) > 1:
        data = data.sum(axis=1) / 2
    frame_length = np.floor(rate / fps)
    index = 0
    # b, a = butter(6, [100 / (rate / 2), 4000 / (rate / 2)], btype='band')
    # data = lfilter(b, a, data)
    data = data[1:] - .95 * data[:-1]
    frame_array = []
    while index < len(data) - frame_length:
        # b = gaussian(39, 10)
        # frame_array.append(wiener(data[index:index + frame_length], 10, .1))
        # frame_array.append(filters.convolve1d(data[index:index + frame_length], b/b.sum()))
        frame_array.append(data[index:index + frame_length])
        if overlap:
            index += np.floor(frame_length / 2)
        else:
            index += frame_length
    samples = Parallel(n_jobs=num_cores, verbose=1)(delayed(get_features_subroutine)(frame, d, rate, frame_length)
                                                    for frame in frame_array)
    keep_columns = list(range(27))
    samples = np.array(samples)[:, keep_columns]
    # delta_1_features = calculate_delta_features(samples, 1)
    delta_features = calculate_delta_features(samples, 2)
    double_delta_features = calculate_delta_features(delta_features, 2)
    all_features = np.hstack((samples[4:-4], delta_features[2:-2], double_delta_features))
    print("%-23s%5.2f%-3s" % ('getFeature Runtime is: ', time.time()-t0, '(s)'))
    return all_features


def calculate_delta_features(features, delta=1):
    delta_features = []
    for i in range(delta, len(features) - delta):
        delta_features.append((features[i + delta] - features[i - delta]) / (2 * delta ** 2))
    return np.array(delta_features)


def get_features_subroutine(frame, d, rate, frame_length):
    energy = np.sum(frame ** 2)
    if np.abs(frame).sum() > 0:
        frame = frame / max(frame)
        frame = frame * np.hamming(frame_length)
        features = mfcc_features(frame, d, rate)
        log_energy = np.log(energy)
    else:
        features = -10 * np.ones(d)
        log_energy = -10
    all_features = np.append(log_energy, features)
    for i, feature in enumerate(all_features):
        if feature > 1e8:
            all_features[i] = 1e4
        elif feature < -1e8:
            all_features[i] = -1e4
        elif np.isnan(feature):
            all_features[i] = 0
    return all_features


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
    minimum_frequency = 300
    maximum_frequency = 3400
    complex_spectrum = np.fft.fft(signal)
    power_spectrum = abs(complex_spectrum) ** 2
    mfb = mel_filter_bank(sample_rate / fps, num_coefficients, minimum_frequency, maximum_frequency)
    filtered_spectrum = np.dot(power_spectrum, mfb)
    log_spectrum = np.log(np.abs(filtered_spectrum) ** 2)
    # log_spectrum = []
    # for value in filtered_spectrum:
    #     if value <= 0:
    #         log_spectrum.append(-10)
    #     else:
    #         log_spectrum.append(np.log(value))
    # log_spectrum = np.array(log_spectrum)
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


def get_subtitles(srt_filename):
    episode_subtitles = pd.DataFrame(columns=['start', 'end', 'text'])
    subs = pysrt.open(srt_filename)
    subs.shift(seconds=0)
    for i, sub in enumerate(subs):
        episode_subtitles.loc[i] = [int(sub.start.minutes) * 60 + int(sub.start.seconds) + sub.start.milliseconds/1000,
                                    int(sub.end.minutes) * 60 + int(sub.end.seconds) + sub.end.milliseconds/1000,
                                    sub.text]
    return episode_subtitles
