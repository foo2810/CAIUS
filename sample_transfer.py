import sys

sys.path.append("./")

import numpy as np
import tensorflow as tf

from utils.datasets import load_data, train_test_split
from utils.common import time_counter
from models.wrapper import VGG16, VGG19

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Params
batch_size = 64
n_epochs = 20
lr = 1e-4
img_size = 128

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=img_size,
                        cache_path='data/data_cache/dataset_size' + str(img_size) + '.pkl')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)
    n_train = len(x_train)
    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(n_train).batch(batch_size)
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

# one-hot vector
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Model
# pretrained model
pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# top
x = Flatten()(pre_model.output)
x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)
y = Dense(3, activation='softmax')(x)

# transferred model
model = Model(inputs=pre_model.input, outputs=y)

# Training
model.compile(optimizer=Adam(lr), loss='categorical_crossentropy',
              metrics=['accuracy'])

stack = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs,
                  validation_data=(x_test, y_test), verbose=1)

score = model.evaluate(x_test, y_test)
print('Val Accuracy: {}'.format(score[1]))
print('Val Loss: {}'.format(score[0]))
