import os
import numpy as np
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
#Seqiential按顺序构成模型
from keras.models import Sequential
#Dense全连接层
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)



def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

#分別將 training set、validation set、testing set 用 readfile 函式讀進來
workspace_dir = './data'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("training图片个数 = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("validation图片个数 = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Testing图片个数 = {}".format(len(test_x)))

print(train_x.shape)
print(train_y.shape)

model = Sequential()

#第一个卷积层
#input_shape 输入平面
#filter 卷积核个数
#kernel_size 卷积窗口大小
#strides 步长
#padding padding方式 same/valid
#activation激活函数
model.add(Convolution2D(input_shape=(128,128,3),filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))

#第一个池化层
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))

#第二个卷积层
model.add(Convolution2D(64,3,strides=1,padding='same',activation='relu'))

#第二个池化层
model.add(MaxPooling2D(2,2,padding='same'))

#第三个卷积层与池化层
model.add(Convolution2D(128,3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2,padding='same'))

#第四个卷积层和池化层
model.add(Convolution2D(256,3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2,padding='same'))

#第五个卷积层与池化层
model.add(Convolution2D(512,3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2,padding='same'))


#将第无个池化层扁平化为一维
model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11,activation='softmax'))
adam=Adam(lr=0.001)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=30)
print("对validation的分类效果为：")
model.evaluate(val_x,val_y)
plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(history.history['loss'],color='r',label='交叉熵损失值')
plt.plot(history.history['accuracy'],color='g',label='精确度')
plt.legend(loc="best")
plt.show()
