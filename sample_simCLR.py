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
from models.wrapper import VGG16, ResNet50, InceptionV3
# from models.wrapper_T import VGG16, ResNet50, InceptionV3

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
batch_size = 64
n_epochs = 2
lr = 0.00001

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128_autopad.pkl', auto_pad_val=True)
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

import time
import pathlib
result_path_raw : str = "results/{}".format(int(time.time()))
result_path : pathlib.Path = pathlib.Path(result_path_raw)
# if not result_path.exists():
#     result_path.mkdir(parents=True)


# Model
# model = SimpleCNN(in_shape, n_out=3)
# model = VGG16(weights=None, classes=3, input_shape=in_shape)
# model = ResNet50(weights=None, classes=3, input_shape=in_shape)
# model = InceptionV3(weights=None, classes=3, input_shape=in_shape)
# model = InceptionV3(classes=3, input_shape=in_shape)
# model = ResNet50(classes=3, input_shape=in_shape)

# =============================================================================
# https://github.com/sayakpaul/SimCLR-in-TensorFlow-2 からのコピペ
# =============================================================================

class CustomAugment(object):
    # Augmentation utilities (differs from the original implementation)
    # Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendxi A 
    # corresponding GitHub: https://github.com/google-research/simclr/)
    def __call__(self, sample):
        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
        
        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    
    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

# Build the augmentation pipeline
data_augmentation = tfk.Sequential([tfk.layers.Lambda(CustomAugment())])

# Architecture utils
def get_resnet_simclr(hidden_1, hidden_2, hidden_3):
    inputs = tfk.Input((128, 128, 3))

    base_model = ResNet50(include_top=False, weights=None)
    base_model.trainabe = True

    h = base_model(inputs, training=True)
    h = tfk.layers.GlobalAveragePooling2D()(h)

    projection_1 = tfk.layers.Dense(hidden_1)(h)
    projection_1 = tfk.layers.Activation("relu")(projection_1)
    projection_2 = tfk.layers.Dense(hidden_2)(projection_1)
    projection_2 = tfk.layers.Activation("relu")(projection_2)
    projection_3 = tfk.layers.Dense(hidden_3)(projection_2)

    resnet_simclr = tfk.Model(inputs, projection_3)

    return resnet_simclr


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)
# Mask to remove positive examples from the batch of negative samples

from utils.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    BATCH_SIZE = xis.shape[0]
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            negative_mask = get_negative_mask(BATCH_SIZE)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def train_simclr(model, dataset, optimizer, criterion, temperature=0.1, epochs=100):
    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in range(epochs):
        for image_batch, _ in dataset:
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        # wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
        
        print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, np.mean(step_wise_loss)))

    return epoch_wise_loss, model


# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr)

# model
resnet_simclr_2 = get_resnet_simclr(256, 128, 50)

# Training
# hist = training(model, train_ds, test_ds, loss, opt, n_epochs, batch_size, weight_name=str(result_path / 'best_param'))
epoch_wise_loss, resnet_simclr  = train_simclr(resnet_simclr_2, train_ds, opt, loss, temperature=0.1, epochs=n_epochs)

# hist_file_path = str(result_path / 'history.csv')
# pd.DataFrame(hist).to_csv(hist_file_path)


# =============================================================================
# Fine-tuning
# =============================================================================

def get_linear_model(features):
    linear_model = tfk.Sequential([tfk.layers.Dense(5, input_shape=(features, ), activation="softmax")])
    return linear_model

# Encoder model with non-linear projections
projection = tfk.Model(resnet_simclr.input, resnet_simclr.layers[-2].output)

# Extract train and test features
train = list(train_ds.unbatch().as_numpy_iterator())
X_train, y_train = map(list, zip(*train))
X_train, y_train = np.array(X_train)[500:], np.array(y_train)[500:]

test = list(test_ds.unbatch().as_numpy_iterator())
X_test, y_test = map(list, zip(*test))
X_test, y_test = np.array(X_test)[500:], np.array(y_test)[500:]

print(np.shape(X_train), np.shape(y_train))
print(np.shape(X_test), np.shape(y_test))

train_features = projection.predict(X_train)
test_features = projection.predict(X_test)

print(train_features.shape, test_features.shape)

linear_model = get_linear_model(128)
linear_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                     optimizer="adam")
history = linear_model.fit(train_features, y_train,
                 validation_data=(test_features, y_test),
                 batch_size=64,
                 epochs=2)