import tensorflow as tf
from keras.optimizers import RMSprop
from keras import layers, models
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten , MaxPooling2D, Dropout
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import ssl

# REALIZED WE WERE DOING THIS ALL WRONG 

# Load CIFAR-10 dataset

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,7))
    p = sns.countplot(y_train.flatten())
    p.set(xticklabels=classes)

    np.isnan(x_train).any()
    np.isnan(x_test).any()

    input_shape = (32, 32, 3)

    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    x_train=x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    x_test=x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    y_train[0]

    plt.imshow(x_train[100])
    print(y_train[100])

    batch_size = 32
    num_classes = 10
    epochs = 50

    model = Sequential([
        Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(64, 3, padding='same', activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-06),
            loss='categorical_crossentropy', metrics=['acc'])
    
    history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs)
    
