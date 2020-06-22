import sys
sys.path.append('./')

import tensorflow as tf
import numpy as np
from utils.datasets import load_data, train_test_split
from utils.common import time_counter
from utils.grad_cam import get_grad_cam, get_grad_cam_plusplus

from models.simple import SimpleCNN
from models.wrapper import ResNet50, VGG16

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfk = tf.keras
# tfk.backend.set_floatx('float32')

# Params
batch_size = 64
n_epochs = 10
lr = 0.001

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=256, cache_path='data/data_cache/dataset_size256_autopad.pkl')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)
    n_train = len(x_train)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(n_train).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_train.shape[1:]
del x_train, y_train, x_test, y_test

# Model
# model = SimpleCNN(in_shape, n_out=3)
# model.load_weights('simplecnn/best_param')
# final_conv_idx = -10
model = VGG16(weights=None, classes=3, input_shape=in_shape)
model.load_weights('vgg16_autopad/best_param')
final_conv_idx = -6
# model = ResNet50(weights=None, classes=3, input_shape=in_shape)
# model.load_weights('resnet50/best_param')
# final_conv_idx = -6

# Training

## Metrics
train_loss = tf.metrics.Mean(name='train_loss')
test_loss = tf.metrics.Mean(name='test_loss')
train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
test_acc = tf.metrics.SparseCategoricalAccuracy(name='test_acc')

## Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# for inputs, labels in train_ds:
#     train_step(model, inputs, labels)

import matplotlib.pyplot as plt
import cv2

gcam_list = []
for label in range(3):
    cnt = 0
    for inputs, labels in test_ds:
        # L, pred = get_grad_cam(model, inputs, label, loss, final_conv_idx)
        L, pred = get_grad_cam_plusplus(model, inputs, label, loss, final_conv_idx)
        pred = np.argmax(pred, axis=1)
        inputs = inputs.numpy()
        labels = labels.numpy()

        width, height = inputs.shape[1:3]
        for org, t, p, gcam in zip(inputs, labels, pred, L):
            gcam = np.uint8(255*gcam)
            resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)
            # resized_gcam = np.transpose(resized_gcam, (1, 0)) # opencvのフォーマットに変換
            org = org*255. # opencvのフォーマットに変換
            resized_gcam = cv2.applyColorMap(resized_gcam, cv2.COLORMAP_JET)
            out = cv2.addWeighted(cv2.cvtColor(org.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, resized_gcam, 0.5, 0)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            plt.clf()
            plt.imshow(out)
            plt.savefig('gcam/{}/id{}_t{}_p{}.png'.format(label, cnt, t, p))
            cnt += 1
