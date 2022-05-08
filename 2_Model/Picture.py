import time, os, glob, cv2, random,os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from Files import *

def to_png(files, name_dir, name_file, librosa=None):
    files = glob(files + '/*.wav')

    for file in range(101):
        data, sr = librosa.load(files[file])
        data = scale(data)

        countstr = str(file)

        melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)

        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        librosa.display.specshow(log_melspec, sr=sr)

        if file < 80:

            # save to png
            directory = name_dir
            if not os.path.exists(directory):
                os.makedirs(directory)

            png_number = name_file + countstr

            plt.savefig(directory + '/' + (png_number) + '.png')
        elif file > 80:

            # save to png
            directory = name_dir + '_validation'
            if not os.path.exists(directory):
                os.makedirs(directory)

            png_number = name_file + countstr

            plt.savefig(directory + '/' + (png_number) + '.png')

    return 0

db = ['-6','0','6']
kinds = ['normal', 'abnormal']
file_name = ['00','02','04','06']