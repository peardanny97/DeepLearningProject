import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D, Flatten, LeakyReLU as Leaky, ReLU,
    BatchNormalization, Dropout,
    MaxPooling2D, Dense
)

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
# for the best performance

from keras.applications import resnet50, MobileNetV2, Xception

# We can use many models from keras, what shoul we use??

train_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Train"
valid_dir = "/Users/bang/Desktop/DeepLearning_Data/CNN_data/Face Mask Dataset/Validation"

datagen = ImageDataGenerator(rescale=1 / 255.)
# will make custom data generator to support image preprocessing
# https://www.kaggle.com/hitzz97/mask-detection-with-mobnetv2

train = datagen.flow_from_directory(train_dir, class_mode='binary', target_size=(160, 160))
test = datagen.flow_from_directory(valid_dir, class_mode='binary', target_size=(160, 160))

train.class_indices
# {'WithMask': 0, 'WithoutMask': 1}
train.batch_size
# 32


# https://www.kaggle.com/taha07/face-mask-detection-using-opencv-mobilenet
# Let's use mobilenet!

mobilenet = MobileNetV2(input_shape=(160, 160, 3), weights="imagenet", include_top=False)
initial_mobile_model = keras.models.Sequential([
    mobilenet,
    Flatten(),
    Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
    Dense(128, activation='relu'),
    Dense(1, activation='softmax'),

])

initial_mobile_model.summary()

initial_mobile_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("/Users/bang/Desktop/DL_Project/DeepLearningProject/CNN/ckpt",
                             monitor="val_accuracy", save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor="val_acc", patience=5, verbose=1)

history = initial_mobile_model.fit(train, epochs=1, validation_data=test, callbacks=[checkpoint, earlystop])

initial_mobile_model.save('/Users/bang/Desktop/DL_Project/DeepLearningProject/CNN/initial_mobile_model')
# initial_model = keras.models.Sequential([
#     Conv2D(input_shape=(256, 256, 3), filters=32, kernel_size=(3,3), padding="Same",
#            activation='relu', kernel_initializer='he_uniform'),
#     Conv2D(32, (3, 3), padding="same"),
#     ReLU(),
#     MaxPooling2D(strides=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.2),
#     Conv2D(64, (3, 3), padding="same"),
#     ReLU(),
#     Conv2D(64, (3, 3), padding="same"),
#     ReLU(),
#     MaxPooling2D(strides=(2, 2)),
#     Conv2D(128, (3, 3), padding="same"),
#     ReLU(),
#     Conv2D(128, (3, 3), padding="same"),
#     ReLU(),
#     MaxPooling2D(strides=(2, 2)),
#     Conv2D(256, (3, 3), padding="same"),
#     Leaky(0.01),
#     Conv2D(256, (3, 3), padding="same"),
#     Leaky(0.01),
#     MaxPooling2D(strides=(2, 2)),
#     Dropout(0.2),
#
#     Flatten(),
#
#     Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
#     Dense(128, activation='relu'),
#     Dense(1, activation='softmax'),
#
# ])

# initial_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss='binary_crossentropy', metrics=['accuracy'])
# initial_model.fit(train, epochs = 1, validation_data=test)
#
# initial_model.save('/Users/bang/Desktop/DL_Project/DeepLearningProject/CNN/initial_model')
