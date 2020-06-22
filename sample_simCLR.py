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
n_epochs = 20
lr = 0.001

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
if not result_path.exists():
    result_path.mkdir(parents=True)


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
def training_SimCLR_Encoder(encoder_model=None, train_ds=None, optimizer=None, n_epochs=20):
    """
    自己教師ありの部分
    """
    
    if train_ds is None:
        raise ValueError("train_ds most not be None")

    if encoder_model is None:
        encoder_model = ResNet50(weights=None, include_top=False)

    if optimizer is None:
        optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
        
    # Image Augmentation
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
    # # TODO 画像拡張の確認
    # import matplotlib.pyplot as plt
    # images, labels = next(iter(train_ds))
    # aug_images = data_augmentation(images)
    # for img_idx in range(5):
    #     print(labels[img_idx])
    #     ax1 = plt.subplot(1, 2, 1)
    #     ax2 = plt.subplot(1, 2, 2)
    #     ax1.imshow(images[img_idx])
    #     ax2.imshow(aug_images[img_idx])
    #     plt.show()
    # sys.exit()

    # Architecture utils
    def get_simclr_model(base: tfk.Model=ResNet50(include_top=False, weights=None), hidden_1=256, hidden_2=128, hidden_3=50):
        inputs = tfk.Input((128, 128, 3))

        base_model = base
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
        epoch_wise_loss = []

        epoch_loss_avg = tfk.metrics.Mean()
        for epoch in range(epochs):
            for image_batch, _ in dataset:
                a = data_augmentation(image_batch)
                b = data_augmentation(image_batch)

                loss = train_step(a, b, model, optimizer, criterion, temperature)
                epoch_loss_avg.update_state(loss)
            
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, epoch_loss_avg.result()))

            epoch_wise_loss.append(epoch_loss_avg.result().numpy())

            epoch_loss_avg.reset_states()

        return epoch_wise_loss, model

    # Loss
    loss = tfk.losses.SparseCategoricalCrossentropy()

    # Optimizer
    opt = tfk.optimizers.Adam(lr)

    # model
    simclr_model = get_simclr_model(encoder_model, 256, 128, 50)

    # Training
    epoch_wise_loss, simclr_model  = train_simclr(simclr_model, train_ds, opt, loss, temperature=0.1, epochs=n_epochs)

    hist = {"'nt_xentloss'": epoch_wise_loss}
    return simclr_model, hist

simclr_model, hist = training_SimCLR_Encoder(train_ds=train_ds, n_epochs=5)

hist_file_path = str(result_path / 'nt_xentloss.csv')
pd.DataFrame(hist).to_csv(hist_file_path)


# =============================================================================
# Fine-tuning
# =============================================================================

# model
num_classes = 3 # TODO Output size
def supervised_model(projection: tfk.Model):
	inputs = tfk.Input((128, 128, 3))
	projection.trainable = False

	r = projection(inputs, training=False)
	outputs = tfk.layers.Dense(num_classes, activation='softmax')(r)

	supervised_model = tfk.Model(inputs, outputs)
  
	return supervised_model

# Encoder model with non-linear projections
projection = tfk.Model(simclr_model.input, simclr_model.layers[-2].output)
linear_model = supervised_model(projection)

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr=lr)

# Training
hist = training(linear_model, train_ds.take(5), test_ds, loss, opt, n_epochs, batch_size, weight_name=str(result_path / 'best_param'))

hist_file_path = str(result_path / 'history.csv')
pd.DataFrame(hist).to_csv(hist_file_path)