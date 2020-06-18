# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:34:25 2020

@author: Dell
"""

import numpy as np
import glob
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image

images_data_Parasitized = []
images_label_Parasitized =[]

images_data_Uninfected = []
images_label_Uninfected =[]

IMG_DIMS = (125,125)

files_Parasitized = glob.glob(r"F:\FYP\FYP\Parasitized/*.PNG")
files_Uninfected = glob.glob(r"F:\FYP\FYP\Uninfected/*.PNG")

#loading PARASITIZED images.
for myFiles_Parasitized in files_Parasitized:
    print(myFiles_Parasitized)
    images_label_Parasitized.append('1')#adding 1 in a label for Parasitized
    image_Parasitized = cv2.imread(myFiles_Parasitized)
    image_Parasitized = cv2.resize(image_Parasitized, dsize=IMG_DIMS,interpolation=cv2.INTER_CUBIC)
    images_data_Parasitized.append(image_Parasitized)
    NpArray_Parasitized = np.array(images_data_Parasitized) # saving image in np.array
    np.asarray(NpArray_Parasitized)

    #akazmi@numl.edu.pk

#loading UNINFECTED images.
for myFiles_Uninfected in files_Uninfected:
    print(myFiles_Uninfected)
    images_label_Uninfected.append('0') #adding 0 in a label for Uninfected
    image_Uninfected = cv2.imread(myFiles_Uninfected)
    image_Uninfected = cv2.resize(image_Uninfected, dsize=IMG_DIMS,interpolation=cv2.INTER_CUBIC) # resize images 125 x 125
    images_data_Uninfected.append(image_Uninfected)
    NpArray_Uninfected = np.array(images_data_Uninfected) # saving image in np.array
    np.asarray(NpArray_Uninfected)
    
#making DATAFRAME and Assigning labels
files_DataFrame = pd.DataFrame({
    'Images': images_data_Parasitized + images_data_Uninfected,
    'Labels': ['Malaria'] * len(images_label_Parasitized) + ['Uninfected'] * len(images_label_Uninfected)
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    
#df = np.array(files_DataFrame)
files_DataFrame.head()
#print(files_DataFrame)
    
#Spliting Images
train_files, test_files, train_labels, test_labels = train_test_split(files_DataFrame['Images'].values,
                                                                      files_DataFrame['Labels'].values, 
                                                                      test_size=0.2)

#print(train_files.shape, val_files.shape, test_files.shape)
#print('Train:', Counter(train_labels), '\nVal:', Counter(val_labels), '\nTest:', Counter(test_labels))  
                                                                    
                                                                    
train_files = train_files / 255
test_files = test_files / 255

train_files = np.array(list(train_files))
test_files = np.array(list(test_files))

le =LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)


EPOCHS = 10
INPUT_SHAPE = (125, 125, 3)

inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), 
                               activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), 
                               activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat = tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary() 

history = model.fit(train_files,train_labels,
                    epochs=EPOCHS,validation_data=(test_files,test_labels)) 

#filename = 'finalized_model'
#pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(test_files, test_labels)
#print(result)
model.save('my_model3.h5',save_format='h5')