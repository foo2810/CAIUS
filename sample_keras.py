import sys
sys.path.append("./")

import numpy as np
import tensorflow as tf

from utils.datasets import load_data, train_test_split
from utils.common import time_counter
from models.wrapper import VGG16
from models.wrapper_T import VGG16 as TransferVGG16
from utils.losses import ComplementEntropy

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Params
batch_size = 128
n_epochs = 100
lr = 1e-4

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset_v2/dataset/', classes=['normal', 'nude', 'swimwear'], size=128,
                        cache_path='data/data_cache/dataset_v2_size128.pkl')
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
model = VGG16(weights=None, classes=3, input_shape=(128, 128, 3))
# model = TransferVGG16(classes=3, input_shape=(128, 128, 3))
# model.layers[0].trainable = False

# Training
model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("weights/vgg16_best_param.hdf5", monitor="val_acc", verbose=1,
                             save_best_only=True, save_weights_only=True)

stack = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs,
                  validation_data=(x_test, y_test), verbose=1, callbacks=[checkpoint])

score = model.evaluate(x_test, y_test)
print('Val Accuracy: {}'.format(score[1]))
print('Val Loss: {}'.format(score[0]))
