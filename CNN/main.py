import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,Flatten,LeakyReLU as Leaky,ReLU,
    BatchNormalization, Dropout,
    MaxPooling2D, Dense
)

train_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Train"
valid_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Validation"

datagen = ImageDataGenerator(rescale=1/255.)
#will make custom data generator to support image resizing
#https://www.kaggle.com/hitzz97/mask-detection-with-mobnetv2

train=datagen.flow_from_directory(train_dir,class_mode='binary')
test=datagen.flow_from_directory(valid_dir,class_mode='binary')

train.class_indices
#{'WithMask': 0, 'WithoutMask': 1}
train.batch_size
#32


initial_model = keras.models.Sequential([
    Conv2D(input_shape=(256, 256, 3), filters=32, kernel_size=(3,3), padding="Same",
           activation='relu', kernel_initializer='he_uniform'),
    Conv2D(32, (3, 3), padding="same"),
    ReLU(),
    MaxPooling2D(strides=(2, 2)),
    BatchNormalization(),
    Dropout(0.2),
    Conv2D(64, (3, 3), padding="same"),
    ReLU(),
    Conv2D(64, (3, 3), padding="same"),
    ReLU(),
    MaxPooling2D(strides=(2, 2)),
    Conv2D(128, (3, 3), padding="same"),
    ReLU(),
    Conv2D(128, (3, 3), padding="same"),
    ReLU(),
    MaxPooling2D(strides=(2, 2)),
    Conv2D(256, (3, 3), padding="same"),
    Leaky(0.01),
    Conv2D(256, (3, 3), padding="same"),
    Leaky(0.01),
    MaxPooling2D(strides=(2, 2)),
    Dropout(0.2),

    Flatten(),

    Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
    Dense(128, activation='relu'),
    Dense(1, activation='softmax'),

])

initial_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy', metrics=['accuracy'])
initial_model.fit(train, epochs = 1, validation_data=test)

initial_model.save('/Users/bang/Desktop/DL_Project/DeepLearningProject/CNN/initial_model')
