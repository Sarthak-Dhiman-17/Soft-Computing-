
import splitfolders
input_folder='data'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output="output", seed=1337, ratio=(.8, .1, .1)) #default values

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras import models

img_width, img_height = 256, 256
train_data_dir = 'output/train'
validation_data_dir = 'output/val'
test_data_dir = 'output/test'

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
       train_data_dir,
       target_size=(img_width, img_height),
       color_mode='grayscale',
       shuffle='True',
       batch_size=32,
       class_mode='categorical',seed=42)

validation_generator = validation_datagen.flow_from_directory(
       validation_data_dir,
       target_size=(img_width, img_height),color_mode='grayscale',
       batch_size=32,
       class_mode='categorical',seed=42)

test_generator = test_datagen.flow_from_directory(
       test_data_dir,
       target_size=(img_width, img_height),color_mode='grayscale',
       batch_size=1,
       class_mode=None,shuffle=False,seed=42)

model = Sequential()
model.add(Convolution2D(8, (3, 3), input_shape=(img_width, img_height,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(8, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
filepath='best_weights.keras'
checkpointer=ModelCheckpoint(filepath,monitor='val_accuracy',mode='max',save_best_only=True,verbose=1)
epochs = 10

history=model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//validation_generator.batch_size)


model.load_weights('best_weights.keras')
model.evaluate(validation_generator)
model.save_weights('cnn_classification.weights.h5')
model.load_weights('cnn_classification.weights.h5')
model.save('cnn_classification.h5')
model = models.load_model('cnn_classification.h5')
import matplotlib.pyplot as plt
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

test_generator.reset()
filenames=test_generator.filenames
nb_samples=len(filenames)
pred=model.predict(test_generator,steps=nb_samples)

predicted_class_indices=np.argmax(pred,axis=1)
labels=(train_generator.class_indices)
labels=dict((v,k) for k,v in labels.items())
predictions=[labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,"Predictions":predictions})
results.to_csv("results.csv",index=False)
