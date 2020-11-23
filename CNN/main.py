import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Train"
test_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Test"

datagen = ImageDataGenerator(rescale=1/255.)
#will make custom data generator to support image resizing
#https://www.kaggle.com/hitzz97/mask-detection-with-mobnetv2

train=datagen.flow_from_directory(train_dir,class_mode='binary')
test=datagen.flow_from_directory(test_dir,class_mode='binary')

train.class_indices
#{'WithMask': 0, 'WithoutMask': 1}
train.batch_size
#32


# base_model = keras.applications.MobileNetV2(
#     weights="imagenet",
#     input_shape=(1024, 1024, 3),
#     include_top=False
# )
#
# inputs = keras.layers.Input(shape=(1024, 1024, 3))
# x = base_model(inputs)
# output = keras.layers.Dense(2)(x)
# model = keras.Model(inputs, output)
#
# model.summary()
# ]