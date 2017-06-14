
# coding: utf-8

# In[169]:

## Trained CIFAR10 
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.models import Model
from keras.models import Sequential

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
#from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator




# In[170]:

plot_history(history)


# In[158]:

(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[159]:

nb_classes = 10
#[0-1]
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0
#one-hot
y_train = keras.utils.to_categorical(y_train,nb_classes)
y_test = keras.utils.to_categorical(y_test,nb_classes)


# In[160]:

#CIFAR10 Model
model = Sequential()

# Block1
model.add(Conv2D(64, (3, 3),activation="relu",padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3),activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

# Block2
model.add(Conv2D(128, (3, 3),activation="relu",padding='same'))
model.add(Conv2D(128, (3, 3),activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

# Block3
model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

# LinearBlock
model.add(Flatten())
model.add(Dense(1024,activation="relu",name="FC1"))
model.add(Dense(1024,activation="relu",name="FC2"))
model.add(Dense(nb_classes,name="output"))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[161]:

batch_size=32
nb_epoch=40
history = model.fit(x_train,y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    validation_data=(x_test,y_test))
model.save("cifar10_epoch40.hdf5")


# In[ ]:




# In[ ]:



