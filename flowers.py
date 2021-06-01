# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:39:36 2021

@author: Hp
"""

#%%
#tensorflow and keras
import tensorflow as tf
from tensorflow import keras

#helper libraries
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
#%%
data="flower_photos/"
folders=os.listdir(data)
print(folders)

#%%
image_names = []
data_labels = []
data_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            data_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            data_images.append(im)
        else:
            continue
        

#%%
data = np.array(data_images)
data.shape

#%%
data=data.astype('float32')/255.0
#%%
label_dummies = pandas.get_dummies(data_labels)

labels =  label_dummies.values.argmax(1)
#%%
pandas.unique(data_labels)
print(pandas.unique(data_labels))
#%%
pandas.unique(labels)
print(pandas.unique(labels))
#%%
# Shuffle the labels and images randomly for better results

union_list = list(zip(data, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,labels,test_size=0.2,random_state=1)
#%%
# Convert the shuffled list to numpy array type

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


#%%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(64,64,3,))) # Input layer
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')) # 2D Convolution layer
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2))) # Max Pool layer 
model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides = (1,1), activation='relu')) # 2D Convolution layer
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2))) # Max Pool layer 
model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides = (1,1), activation='relu')) # 2D Convolution layer
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2))) # Max Pool layer 
model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides = (1,1), activation='relu')) # 2D Convolution layer
model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2))) # Max Pool layer 
model.add(tf.keras.layers.GlobalMaxPool2D()) # Global Max Pool layer
model.add(tf.keras.layers.Flatten()) # Dense Layers after flattening the data
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2)) # Dropout
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.BatchNormalization()) # Normalization layer
model.add(tf.keras.layers.Dense(5, activation='softmax')) # Add Output Layer
#%%
# Compute the model parameters

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#%%
model.fit(x_train,y_train, epochs=20)
#%%
y_pred=model.predict(x_test)
#%%
y_pred = np.argmax(y_pred,axis=1)


#%%
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)
sn.heatmap(cm, annot=True, fmt="d")
plt.show()
#%%
#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))


#%%
sunflower_url = "sun.jpg"
#sunflower_path = tf.keras.utils.get_file('download.jpg')

img = keras.preprocessing.image.load_img(
    sunflower_url, target_size=(64, 64)
)
img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print(folders[np.argmax(score)])
print(
    "bu goruntu {} sınıfına ait, {:.2f}  confidence deger ile."
    .format(folders[np.argmax(score)], 100 * np.max(score))
)

print(score)
print(folders[np.argmax(score)])
#%%
#%%
from tensorflow.keras.models import model_from_yaml
# serialize model to YAML
#%%
model_yaml = model.to_yaml()
with open("model3.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model3.h5")
print("Saved model to disk")
 



















