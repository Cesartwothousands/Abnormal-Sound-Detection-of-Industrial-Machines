import numpy as np
import matplotlib.pyplot as plt
import glob, os, random, time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD

# print(tf.__version__)

time_start = time.time()  # time = 0

root_path = r'F:\毕业论文'
labels = {1: 'normal', 0: 'abnormal'}

rate = 0.3
batchsize = 10
classmode = 'binary'
picsize = (1000,800)
inputshape = (1000,800,3)

# 构造ImageDataGenerator对象
train_datagen = ImageDataGenerator(
    rescale=1. / 255,                   # 数据缩放，把像素点的值除以255，使之在0到1之间
    shear_range=0.1,                    # 错切变换角度
    zoom_range=0.1,                     # 随机缩放范围
    width_shift_range = 0.1,            # 除以总宽度的值
    height_shift_range = 0.1,           # 除以总高度的值
    horizontal_flip=True,               # 随机水平暗转
    vertical_flip = True,               # 随机垂直翻转
    validation_split = rate             # 保留用于验证的图像比例为0.3
)

# 归一化验证集
val_datagen = ImageDataGenerator(
    rescale=1. / 255,                  # 数据缩放，把像素点的值除以255，使之在0到1之间
    validation_split= rate)              # 保留用于验证的图像比例为0.3

# 对训练集进行数据增强
train_generator = train_datagen.flow_from_directory(
    root_path+'\Model Data',              # 目标数据的路径
    target_size= picsize,            # 所有的图片将被调整的尺寸
    batch_size=batchsize,                     # 每批传入数据的大小
    class_mode= classmode,          # 决定返回标签的类型。这里采用 2D one-hot 编码标签
    subset='training',                 # 数据子集
    seed=None)                            # 可选随机种子

# 对测试集进行数据增强
val_generator = val_datagen.flow_from_directory(
    root_path+'\Model Data',              # 目标数据的路径
    target_size= picsize,            # 所有的图片将被调整的尺寸
    batch_size=batchsize,                     # 每批传入数据的大小
    class_mode= classmode,          # 决定返回标签的类型。这里采用 2D cone-hot 编码标签
    subset='validation',               # 数据子集
    seed=None)                            # 可选随机种子

### model
model = Sequential()
# 卷积层
# filter=32表示卷积滤波器个数为32
# kernel_size=3表示所有空间维度指定相同的值
# padding='same'表示padding完尺寸与原来相同
# activation='relu'使用relu作为激活函数
# input_shape=(300, 300, 3)表示输入图像的尺寸为300x300，且有3个颜色通道

#1
model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape= inputshape))
# 池化层
# 使用最大池化层，且最大池化的窗口为2
model.add(MaxPooling2D(pool_size=3))
#2
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
#3
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
#4
model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
#5
model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())  # 扁平化参数
# 全连接层
# 全连接层输出的空间维度为64
# 激活函数采用relu
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
# 全连接层
# 全连接层输出的空间维度为6
# 激活函数采用softmax
model.add(Dense(1, activation='sigmoid'))
# 完成架构搭建后，最后输出模型汇总

from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

# Build Model...


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),           # 损失函数使用交叉熵
                   optimizer=tf.keras.optimizers.Adam(lr = 0.001),                     # 设置优化器
                   metrics=['accuracy',auroc])                 # 设置评估指标为准确率

### train
time_model = time.time() # 记录训练开始时间
history_fit = model.fit(train_generator,                # 增强的数据集
                        epochs=100,                      # 迭代总轮数，这里设置为50次，你可以在实验是增加epoch次数，提升准确率
                        steps_per_epoch=int(len(train_generator)//batchsize),       # generator 产生的总步数（批次样本）
                        validation_data=val_generator,  # 验证数据的生成器
                        validation_steps=int(len(val_generator)//batchsize)         # 在停止前 generator 生成的总步数（样本批数）
                        )

model.summary()

########################################################################
time_end = time.time()
minute = (time_end - time_model) // 60
second = (time_end - time_model) % 60
print('\nModel Time cost', minute, 'min ', second, ' sec')


