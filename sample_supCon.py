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
from utils.common import time_counter

from models.simple import SimpleCNN
from models.wrapper import VGG16, ResNet50
from utils.losses import SupConLoss

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
# =============================================================================
# https://github.com/sayakpaul/Supervised-Contrastive-Learning-in-TensorFlow-2 からのコピペ
# =============================================================================

def training_SupCon_Encoder(encoder_model=None, train_ds=None, optimizer=None, n_epochs=20):
    """
    Encoder nework and Projection network 自己教師ありの部分
    """

    if train_ds is None:
        raise ValueError("train_ds most not be None")

    if encoder_model is None:
        encoder_model = ResNet50(weights=None, include_top=False)

    if optimizer is None:
        optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
        

    class UnitNormLayer(tf.keras.layers.Layer):
        # Reference: https://github.com/wangz10/contrastive_loss/blob/master/model.py
        '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
        '''
        def __init__(self):
            super().__init__()

        def call(self, input_tensor):
            norm = tf.norm(input_tensor, axis=1)
            return input_tensor / tf.reshape(norm, [-1, 1])
        
    # Encoder Network
    def encoder_net(encoder_model=ResNet50(weights=None, include_top=False)):
        inputs = tfk.Input((128, 128, 3))
        normalization_layer = UnitNormLayer()

        encoder = encoder_model
        encoder.trainable = True

        embeddings = encoder(inputs, training=True)
        embeddings = tfk.layers.GlobalAveragePooling2D()(embeddings)
        norm_embeddings = normalization_layer(embeddings)

        encoder_network = tfk.Model(inputs, norm_embeddings)

        return encoder_network

    # Projector Network
    def projector_net(encoder_r: tfk.Model):
        encoder_r.trainable = True
        projector = tfk.models.Sequential([
            encoder_r,
            tfk.layers.Dense(256, activation="relu"),
            UnitNormLayer()
        ])

        return projector

    # Training the encoder and the projector

    encoder_r = encoder_net(encoder_model)
    projector_z = projector_net(encoder_r)
    supConloss = SupConLoss()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            z = projector_z(images, training=True)
            loss = supConloss(z, labels)

        gradients = tape.gradient(loss, projector_z.trainable_variables)
        optimizer.apply_gradients(zip(gradients, projector_z.trainable_variables))

        return loss

    train_loss_results = []

    epoch_loss_avg = tf.keras.metrics.Mean()
    for epoch in range(n_epochs):	
        for (images, labels) in train_ds:
            loss = train_step(images, labels)
            epoch_loss_avg.update_state(loss) 

            # TODO loss が nan になることがある
            if tf.math.is_nan(loss):
                print("loss", loss)
                print("encoder_r output is nan:", tf.reduce_any(tf.math.is_nan(encoder_r(images))))
                print("projector_z output is nan:", tf.reduce_any(tf.math.is_nan(projector_z(images))))
                sys.exit()

        print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, n_epochs, epoch_loss_avg.result()))

        train_loss_results.append(epoch_loss_avg.result().numpy())

        epoch_loss_avg.reset_states()

    hist = {"'supervised_contrastive_loss'": train_loss_results}
    return encoder_r, hist

# SGD with lr decay function
optimizer = tfk.optimizers.Adam(learning_rate=1e-3)

encoder_r, hist = training_SupCon_Encoder(train_ds=train_ds, optimizer=optimizer, n_epochs=15)

supCon_file_path = str(result_path / 'supCon.csv')
pd.DataFrame(hist).to_csv(supCon_file_path)

# =============================================================================
# Classifer 部分　教師あり学習
# =============================================================================

# model
num_classes = 3 # TODO Output size
def supervised_model(encoder_r: tfk.Model):
	inputs = tfk.Input((128, 128, 3))
	encoder_r.trainable = False

	r = encoder_r(inputs, training=False)
	outputs = tfk.layers.Dense(num_classes, activation='softmax')(r)

	supervised_model = tfk.Model(inputs, outputs)
  
	return supervised_model

model = supervised_model(encoder_r)

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr=lr)

# Training
hist = training(model, train_ds, test_ds, loss, opt, n_epochs, batch_size, weight_name=str(result_path / 'best_param'))

hist_file_path = str(result_path / 'history.csv')
pd.DataFrame(hist).to_csv(hist_file_path)
    