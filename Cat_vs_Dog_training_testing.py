# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 00:06:38 2021

@author: Krishna Sharma
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from random import shuffle
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "PetImages.zip"
  
# Extracting all content of the zip file
with ZipFile(file_name, 'r') as zip:
    zip.extractall()


#Image Labeling. One-hot encoding was done just by a simple function. 
def image_label(element):
    if element=="Dog":
        return [0,1]
    if element=="Cat":
        return [1,0]

#Defining the class elements    
classes={"Cat","Dog"}

#The image data from the folder is read by this code. 
data=[]
for class_element in classes:
    directory="PetImages\\"+class_element  #Setting the image path from this line
    print(class_element)
    for img in list(os.listdir(directory)):
        label = image_label(class_element)
        path =  os.path.join(directory,img)
        image = cv2.imread(path) #Reading the image by open_cv library
        try:
            image=cv2.resize(image,(150,150)) #resizing the the image. It is reduced for the capacity of the machine
            data.append([np.array(image),np.array(label)]) #appending the input and output of the data
        except:
            pass

#Suffle it to increase the randomness of the data
shuffle(data)   

#Separating input data and output label
X=[]
y=[]
for feature,label in data:
    X.append(feature)
    y.append(label)
X=np.asarray(X)
y=np.asarray(y)
X=X/255
print(X.shape)
print(y.shape) 

#splitting the data as train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
#Defining the CNN architecture. You can use any combination which is suitable for the data
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
print(model.summary())   

#Compliling the model with some hyper-parameter    
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

#Start training the model
history = model.fit(X_train, y_train, batch_size=8, epochs=100, validation_split=0.2)

#predicting the output
y_pred=model.predict(X_test) 

#As the label is one-hot encoded, We have to convert it as single value
y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

#printing the classification report
report=classification_report(y_test,y_pred)
print(report)

#printing the confusion marix
confusion_report=confusion_matrix(y_test,y_pred)
print(confusion_report)


