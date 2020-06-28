import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

from utils.datasets import load_data, train_test_split
from utils.train import training_supCon
from utils.data_augment import random_flip_left_right, random_rotate_90, gen_random_cutout
from utils.common import time_counter

from models.wrapper import DenseNet121
# from models.wrapper_T import VGG16, ResNet50, InceptionV3

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
n_classes = 3
batch_size = 64
n_epochs = 10
lr = 5.8e-5

# Dataset
print('[Dataset]')
with time_counter():
    with open('train_size128.pkl', 'rb') as fp:
        x_train, y_train = pickle.load(fp)
    with open('test_size128.pkl', 'rb') as fp:
        x_test, y_test = pickle.load(fp)
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

import time
import pathlib
result_path_raw : str = "data/data_cache/{}".format(int(time.time()))
result_path : pathlib.Path = pathlib.Path(result_path_raw)
if not result_path.exists():
    result_path.mkdir(parents=True)


# Model
model = DenseNet121(weights=None, include_top=False)

# Loss
loss = tfk.losses.CategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr)

# Training
# pre_w = model.layers[-4].weights[0].numpy().copy()
# train_ds = train_ds.take(2)
hist = training_supCon(model, train_ds, test_ds, loss, opt, n_epochs, batch_size, 
                        n_classes, alpha=0.2, weight_name=str(result_path / 'best_param'),
                        encoder_opt=tfk.optimizers.Adam(2e-4), encoder_epochs=10)
# post_w = model.layers[-4].weights[0].numpy()
# print(np.array_equal(pre_w, post_w))

hist_file_path = str(result_path / 'history.csv')
pd.DataFrame(hist).to_csv(hist_file_path)