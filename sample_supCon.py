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

# =============================================================================
# Encoder nework and Projection network 自己教師ありの部分
# =============================================================================
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
def encoder_net():
	inputs = tfk.Input((128, 128, 3))
	normalization_layer = UnitNormLayer()

	encoder = ResNet50(weights=None, include_top=False)
	encoder.trainable = True

	embeddings = encoder(inputs, training=True)
	embeddings = tfk.layers.GlobalAveragePooling2D()(embeddings)
	norm_embeddings = normalization_layer(embeddings)

	encoder_network = tfk.Model(inputs, norm_embeddings)

	return encoder_network

# Projector Network
def projector_net():
	projector = tfk.models.Sequential([
		tfk.layers.Dense(256, activation="relu"),
		UnitNormLayer()
	])

	return projector

# Training the encoder and the projector

# SGD with lr decay function
optimizer = tfk.optimizers.Adam(learning_rate=1e-3)

encoder_r = encoder_net()
projector_z = projector_net()

supConloss = SupConLoss()

@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		r = encoder_r(images, training=True)
		z = projector_z(r, training=True)
		loss = supConloss(z, labels)

	gradients = tape.gradient(loss, 
		encoder_r.trainable_variables + projector_z.trainable_variables)
	optimizer.apply_gradients(zip(gradients, 
		encoder_r.trainable_variables + projector_z.trainable_variables))

	return loss

EPOCHS = 20
LOG_EVERY = 1
train_loss_results = []

print('[training representations]')
with time_counter():
    for epoch in range(EPOCHS):	
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        for (images, labels) in train_ds:
            loss = train_step(images, labels)
            epoch_loss_avg.update_state(loss) 

            # TODO loss が nan になることがある
            if tf.math.is_nan(loss):
                print("loss", loss)
                print("encoder_r output is nan:", tf.reduce_any(tf.math.is_nan(encoder_r(images))))
                print("projector_z output is nan:", tf.reduce_any(tf.math.is_nan(projector_z(encoder_r(images)))))
                sys.exit()

        train_loss_results.append(epoch_loss_avg.result())
        # print({"supervised_contrastive_loss": epoch_loss_avg.result()})

        if epoch % LOG_EVERY == 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

supCon_file_path = str(result_path / 'supCon.csv')
pd.DataFrame(train_loss_results, columns=['supervised_contrastive_loss']).to_csv(supCon_file_path)


# =============================================================================
# Classifer 部分　教師あり学習
# =============================================================================

# model
num_classes = 3 # TODO Output size
def supervised_model():
	inputs = tfk.Input((128, 128, 3))
	encoder_r.trainable = False

	r = encoder_r(inputs, training=False)
	outputs = tfk.layers.Dense(num_classes, activation='softmax')(r)

	supervised_model = tfk.Model(inputs, outputs)
  
	return supervised_model

model = supervised_model()

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr=lr)

# Training
hist = training(model, train_ds, test_ds, loss, opt, n_epochs, batch_size, weight_name=str(result_path / 'best_param'))

hist_file_path = str(result_path / 'history.csv')
pd.DataFrame(hist).to_csv(hist_file_path)
    