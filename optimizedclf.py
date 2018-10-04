import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from tensorflow.keras.callbacks import TensorBoard
import time


#NAME = 'Cats-vs-dog-cnn-64x2-{}'.format(int(time.time()))



DATADIR = 'C:/datasets/train' #my datadir, download yours at https://www.kaggle.com/c/dogs-vs-cats/data and choose your prefered dir
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 100

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to cats dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X/255.0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.555)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):  
                model.add( Conv2D(layer_size, (3,3)) )
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])
