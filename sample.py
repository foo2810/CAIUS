import sys
sys.path.append('./')

import tensorflow as tf
import numpy as np
import pandas as pd

from utils.datasets import load_data, train_test_split
from utils.common import time_counter

from models.simple import SimpleCNN
from models.wrapper import VGG16, ResNet50

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
batch_size = 64
n_epochs = 20
lr = 0.001

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=256, cache_path='data/data_cache/dataset_size256_autopad.pkl', auto_pad_val=True)
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
# model = VGG16(weights=None, classes=3, input_shape=in_shape)
# model = ResNet50(weights=None, classes=3, input_shape=in_shape)

# Training

## Metrics
train_loss = tf.metrics.Mean(name='train_loss')
test_loss = tf.metrics.Mean(name='test_loss')
train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
test_acc = tf.metrics.SparseCategoricalAccuracy(name='test_acc')

## Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

## Optimizer
opt = tfk.optimizers.Adam(lr=lr)

@tf.function
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss_val = loss(labels, pred)
    grads = tape.gradient(loss_val, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss_val)
    train_acc(labels, pred)

@tf.function
def test_step(model, inputs, labels):
    pred = model(inputs)
    loss_val = loss(labels, pred)
    test_loss(loss_val)
    test_acc(labels, pred)

hist = {
    'train_loss': [],
    'test_loss': [],
    'train_acc': [],
    'test_acc': [],
}
template = 'Epoch[{}/{}] loss: {:.3f}, acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}'
best_acc = 0
for epoch in range(n_epochs):
    for inputs, labels in train_ds:
        train_step(model, inputs, labels)
    
    for inputs, labels in test_ds:
        test_step(model, inputs, labels)
    
    if best_acc < test_acc.result().numpy():
        best_acc = test_acc.result().numpy()
        model.save_weights('best_param', save_format='tf')
    
    print(template.format(
        epoch+1, n_epochs,
        train_loss.result().numpy(),
        train_acc.result().numpy(),
        test_loss.result().numpy(),
        test_acc.result().numpy(),
    ))

    hist['train_loss'] += [train_loss.result().numpy()]
    hist['test_loss'] += [test_loss.result().numpy()]
    hist['train_acc'] += [train_acc.result().numpy()]
    hist['test_acc'] += [test_acc.result().numpy()]
    
    train_loss.reset_states()
    test_loss.reset_states()
    train_acc.reset_states()
    test_acc.reset_states()


pd.DataFrame(hist).to_csv('history.csv')
    