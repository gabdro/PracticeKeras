
# coding: utf-8

# In[169]:

## Trained CIFAR10 
import numpy as np
#import matplotlib.pyplot as plt

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

#plot_history(history)


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

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=0,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=False)
datagen.fit(x_train)

# In[160]:

#CIFAR10 Model
model = Sequential()

# Block1
model.add(Conv2D(32, (3, 3),activation="relu",padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3),activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),activation="relu",padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3),activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

# Block2
#model.add(Conv2D(128, (3, 3),activation="relu",padding='same'))
#model.add(Conv2D(128, (3, 3),activation="relu",padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.25))

# Block3
#model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
#model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
#model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
#model.add(Conv2D(256,(3,3),activation="relu",padding='same',))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.25))

# LinearBlock
model.add(Flatten())
model.add(Dense(1024,activation="relu",name="FC1"))
model.add(Dense(1024,activation="relu",name="FC2"))
model.add(Dense(nb_classes,name="output"))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#model.summary()

from keras.models import load_model
model = load_model("now.hdf5")
model.summary()

# In[161]:

batch_size=32
nb_epoch=200
#history = model.fit(x_train,y_train,
#                    batch_size=batch_size,
#                    epochs=nb_epoch,
#                    validation_data=(x_test,y_test))
#model.save("cifar10_epoch40.hdf5")

#MELMEL
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="../MODEL/{epoch:2d}_cifar10_aug.hdf5",save_best_only=True,verbose=1)

model.fit_generator(datagen.flow(x_train,y_train,
                                 batch_size=batch_size),
                                 samples_per_epoch=x_train.shape[0] // batch_size,
                                 epochs=nb_epoch,
                                 validation_data=(x_test,y_test),
                                 callbacks=[checkpoint])
#model.save("augmentation_cifar10_epoch40.hdf5")
# In[ ]:




# In[ ]:



