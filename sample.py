import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd

from utils.datasets import load_data, train_test_split
from utils.train import training
from utils.data_augment import random_flip_left_right, random_rotate_90, gen_random_cutout
from utils.common import time_counter

from models.simple import SimpleCNN
#from models.wrapper import VGG16, ResNet50, InceptionV3
from models.wrapper_T import VGG16, ResNet50, InceptionV3

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
batch_size = 64
n_epochs = 50
lr = 0.00001

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128_autopad.pkl', auto_pad_val=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)
    n_train = len(x_train)

    _random_cutout = gen_random_cutout(42)
    @tf.function
    def augment(image, label):
        image, label = random_flip_left_right(image, label)
        # image, label = random_flip_up_down(image, label)
        image, label =_random_cutout(image, label)
        image, label = random_rotate_90(image, label)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(n_train).map(augment) \
        .batch(batch_size).repeat(3)    # datasetのサイズを3倍に
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_train.shape[1:]
del x_train, y_train, x_test, y_test

# Model
# model = SimpleCNN(in_shape, n_out=3)
# model = VGG16(weights=None, classes=3, input_shape=in_shape)
# model = ResNet50(weights=None, classes=3, input_shape=in_shape)
# model = InceptionV3(weights=None, classes=3, input_shape=in_shape)
# model = InceptionV3(classes=3, input_shape=in_shape)
model = ResNet50(classes=3, input_shape=in_shape)

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr)

# Training
hist = training(model, train_ds, test_ds, loss, opt, n_epochs, batch_size)

pd.DataFrame(hist).to_csv('history.csv')
    