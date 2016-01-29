###Standard Python packages###
import subprocess
import os
import re
import sys
###Other###
from tkinter import Tk, filedialog

__author__ = 'Brandon'


called_version = sys.version_info


def directory_avi_to_wav(path=None):
    """
    Convert all avi files in a directory to wav. Saves in subdirectory of input path
    :param path: the path to the folder containing avi files
    :return:
    """
    # If no input path, use GUI
    if path is None:
        path = get_path()
    # Rename the avi files for easier use
    wav_subdirectory_path = os.path.join(path, 'wav_files')
    rename_files(path)
    # Iterate through all files
    for filename in os.listdir(path):
        # Split path by extension
        file_path, extension = os.path.splitext(filename)
        print(file_path, extension)
        # Process only if file is avi
        if extension == '.avi':
            if not os.path.exists(wav_subdirectory_path):
                os.makedirs(wav_subdirectory_path)
            numbers = re.findall(r'\d+', filename)
            wav_file = "Simpsons_" + numbers[0] + "x" + numbers[1] + ".wav"
            print('wav_file: ', os.path.join(wav_subdirectory_path, wav_file))
            avi_to_wav(filename, os.path.join(wav_subdirectory_path, wav_file))


def rename_files(path):
    """
    Rename files to remove whitespace
    :param path: the path to the folder containing avi files
    :return:
    """
    for filename in os.listdir(path):
        os.rename(os.path.join(path, filename), os.path.join(path, '_'.join(filename.split())))


def path_wav_to_wav(path=None):
    """
    This converts WAV files to specific audio setting
    :param path: the path to the folder containing wav files
    :return:
    """
    if path is None:
        path = get_path()
    for filename in os.listdir(path):
        file_path, extension = os.path.splitext(filename)
        if extension == 'wav':
            wav_file = file_path+'_clean.wav'
            wav_to_wav(os.path.join(path, filename), os.path.join(path, wav_file))


def avi_to_wav(avi_file, wav_file):
    """
    Converts AVI video file to WAV audio file with specific audio setting
    :param avi_file: path to
    :param wav_file:
    :return:
    """
    # Executable program on Windows for AV editing
    ffmpeg = os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin', 'ffmpeg.exe')
    print(ffmpeg)
    command = ffmpeg + " -i " + avi_file + " -ab 160k -ac 1 -ar 10000 -vn " + wav_file
    subprocess.call(command, shell=True)


def wav_to_wav(input_wav_file, output_wav_file):
    ffmpeg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg', 'bin', 'ffmpeg.exe')
    command = ffmpeg + " -i " + input_wav_file + " -ac 1 -ar 10000 -vn " + output_wav_file
    subprocess.call(command, shell=True)


def get_path():
    """GUI for selecting the proper CSV files only works in Python 3
    Inputs: None
    :return: str, filename output based on user selection
    """
    ###File Selection GUI###
    root = Tk()
    path = filedialog.askdirectory(title='Find path for Episodes')
    root.destroy()
    #################
    return path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_avi_to_wav(sys.argv[1])
    else:
        directory_avi_to_wav()
