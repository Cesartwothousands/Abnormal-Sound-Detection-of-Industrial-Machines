import time, os, glob, cv2, random,os
import matplotlib.pyplot as plt
import numpy as np
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

time_start = time.time()  # time = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# test GPU available: tf.config.list_physical_devices('GPU')

labels = ["abnormal", "normal"]
img_size = 400


def get_data(pic):
    data=[]

    for label in labels:
        path = os.path.join(pic, label)   # create path
        class_num = labels.index(label)   # get the classification  (0 or a 1). 0=Abnormal 1=normal
        for img in os.listdir(path):      # iterate over each image per two of them
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size+600))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except:
                print('error in data')
    return np.array(data,dtype=object)

train = get_data(r'F:\Graduate_project\Pictures\Mel\fan')



time_model = time.time()
########################################################################
time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nModel Time cost', minute, 'min ', second, ' sec')