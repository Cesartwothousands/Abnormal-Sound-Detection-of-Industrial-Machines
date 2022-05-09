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
from RaTe import RaTe

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# test GPU available: tf.config.list_physical_devices('GPU')

labels = ["abnormal", "normal"]
img_size = 400
rate = 0.2

machines = ['fan','pump','slider','valve']
kinds = ['normal', 'abnormal']
rootpath = f'F:/Graduate_projrct/Pictures/Mel/'
def name_path(name):
    paths = []
    for machine in machines:
        paths.append(rootpath+f'{machine}/{name}')
    return paths
file_names = {'normal': name_path(kinds[0]),'abnormal': name_path(kinds[1])}

def RaTe(num,rate):
    t = int(num * rate)
    R = []
    while(1):
        for i in range(num):
            if t == 0:
                R.append(0)
            else:
                r = random.random()
                if r <= rate:
                    R.append(1)
                    t -= 1
                else:
                    R.append(0)
        if t == 0:
            break
        else:
            continue
    return R

def get_data(pic):
    train_data=[]
    test_data=[]

    for label in labels:
        path = os.path.join(pic, label)   # create path
        class_num = labels.index(label)   # get the classification  (0 or a 1). 0=Abnormal 1=normal
        for img in os.listdir(path):      # iterate over each image per two of them
            l = len(os.listdir(path))
            R = RaTe(l, rate)

            for file in range(l):
                if R[file] == 0:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                    resized_arr = cv2.resize(img_arr, (img_size, img_size + 600))  # Reshaping images to preferred size
                    train_data.append([resized_arr, class_num])
                else:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                    resized_arr = cv2.resize(img_arr, (img_size, img_size + 600))  # Reshaping images to preferred size
                    test_data.append([resized_arr, class_num])

    Result=[]
    Result.append(train_data)
    Result.append(test_data)
    return Result

Result = get_data(r'F:\Graduate_project\Pictures\Mel\fan')
train = np.array(Result[0],dtype=object)
test = np.array(Result[1],dtype=object)

print(train[0,0].shape)
print(test[0,0].shape)
print('Train len',len(train))
print('Test len',len(test))

x_train = []
y_train = []
x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
# Normalize the data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

###model###model###model###model###model###model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs = 10 , validation_data = (x_test, y_test))

time_start = time.time()  # time = 0
time_model = time.time()
########################################################################
time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nModel Time cost', minute, 'min ', second, ' sec')