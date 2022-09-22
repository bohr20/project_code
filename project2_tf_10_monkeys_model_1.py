#!/usr/bin/env python
# coding: utf-8

# In[17]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


# In[19]:


train_dir = "../input/training"
valid_dir = "../input/validation"
label_file = "../input/monkey_labels.txt"
print(os.path.exists(train_dir))
print(os.path.exists(valid_dir))
print(os.path.exists(label_file))

print(os.listdir(train_dir))
print(os.listdir(valid_dir))


# In[20]:


labels = pd.read_csv(label_file, header=0)
print(labels)


# In[21]:


height = 128
width = 128
channels = 3
batch_size = 64
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,#图片中的每个像素点都乘以1/255
    rotation_range = 40,#把图形随机旋转一个角度，在0-40度之间
    width_shift_range = 0.2,#位移，0-20%之间选择做偏移
    height_shift_range = 0.2,#垂直方向位移，如果是0-1之间的数，就是比例，大于1就是像素
    shear_range = 0.2,#剪切强度（逆时针剪切角，以度为单位）
    zoom_range = 0.2,#缩放强度
    horizontal_flip = True,#水平随机翻转
    fill_mode = 'nearest',#图形放大后，有些地方需要填充，
#  可以是“常数”，“最近”，“反射”或“环绕”之一。默认值为“最近”。输入边界之外的点将根据给定模式进行填充
)
#读取图片
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    #生成图片多大
                                                   target_size = (height, width),
                                                    #生成的图片以多少张为一组
                                                   batch_size = batch_size,
                                                   seed = 7,
                                                   shuffle = True,
                                                    #one-hot编码后的一种模式
                                                   class_mode = "categorical")
#对验证集必须做一个值的缩放，其他的不需要做
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size = (height, width),
                                                    batch_size = batch_size,
                                                    seed = 7,
                                                    shuffle = False,
                                                    class_mode = "categorical")
train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)


# In[22]:


for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    print(y)


# In[23]:


model = keras.models.Sequential([
    #第一组卷积
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu', input_shape=[width, height, channels]),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    #第二组翻倍
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    #第三组再翻倍
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    #展平后，和全连接层做连接
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])
model.summary()


# In[24]:


len(model.layers)


# In[15]:


epochs = 10
#如果机器性能好，可以把epochs改为300
history = model.fit_generator(train_generator,
                              #显试指定
                              steps_per_epoch = train_num // batch_size,
                              epochs = epochs,
                              validation_data = valid_generator,
                              validation_steps = valid_num // batch_size)


# In[16]:


def plot_learning_curves(history, label, epcohs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()
#因为accuracy的值和loss值的范围很不一样，因此我们打印两条曲线
plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 1.5, 2.5)


# In[ ]:




