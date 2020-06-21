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

# =============================================================================
# define max_margin_contrastive_loss
# 以下のファイルからコピペ
# tensorflow_addons, https://raw.githubusercontent.com/wangz10/contrastive_loss/master/losses.py
# =============================================================================
def pdist_euclidean(A):
    # Euclidean pdist
    # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return tf.sqrt(D)

def square_to_vec(D):
    '''Convert a squared form pdist matrix to vector form.
    '''
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_vec = tf.gather_nd(D, list(zip(triu_idx[0], triu_idx[1])))
    return d_vec

def get_contrast_batch_labels(y):
    '''
    Make contrast labels by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    D_y = pdist_euclidean(y_col_vec)
    d_y = square_to_vec(D_y)
    y_contrasts = tf.cast(d_y == 0, tf.int32)
    return y_contrasts

from typing import Union, List

import numpy as np
import tensorflow as tf

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
]
@tf.function
def contrastive_loss(
    y_true: TensorLike, y_pred: TensorLike, margin: Number = 1.0
) -> tf.Tensor:
    r"""Computes the contrastive loss between `y_true` and `y_pred`.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape [batch_size, hidden_size] can be computed
    as follows:
    ```python
    # y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
    y_pred = tf.linalg.norm(a - b, axis=1)
    ```
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape [batch_size] of
        distances between two embedding matrices.
      margin: margin term in the loss definition.
    Returns:
      contrastive_loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )

def max_margin_contrastive_loss(z, y, margin=1.0, metric='euclidean'):
    '''
    Wrapper for the maximum margin contrastive loss (Hadsell et al. 2006)
    `tfa.losses.contrastive_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
        metric: one of ('euclidean', 'cosine')
    '''
    # compute pair-wise distance matrix
    if metric == 'euclidean':
        D = pdist_euclidean(z)
    elif metric == 'cosine':
        D = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
    # convert squareform matrix to vector form
    d_vec = square_to_vec(D)
    # make contrastive labels
    y_contrasts = get_contrast_batch_labels(y)
    loss = contrastive_loss(y_contrasts, d_vec, margin=margin)
    # exploding/varnishing gradients on large batch?
    return tf.reduce_mean(loss)
# ------

@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		r = encoder_r(images, training=True)
		z = projector_z(r, training=True)
		loss = max_margin_contrastive_loss(z, labels)

	gradients = tape.gradient(loss, 
		encoder_r.trainable_variables + projector_z.trainable_variables)
	optimizer.apply_gradients(zip(gradients, 
		encoder_r.trainable_variables + projector_z.trainable_variables))

	return loss

EPOCHS = 15
LOG_EVERY = 1
train_loss_results = []

start = time.time()
for epoch in range(EPOCHS):	
	epoch_loss_avg = tf.keras.metrics.Mean()
	
	for (images, labels) in train_ds:
		loss = train_step(images, labels)
		epoch_loss_avg.update_state(loss) 

	train_loss_results.append(epoch_loss_avg.result())
	# print({"supervised_contrastive_loss": epoch_loss_avg.result()})

	if epoch % LOG_EVERY == 0:
		print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

end = time.time()
print({"training_time": end - start})


# import matplotlib.pyplot as plt
# plt.plot(train_loss_results)
# plt.title("Supervised Contrastive Loss")
# plt.show()
# # print(train_loss_results)
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
    