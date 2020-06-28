import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from utils.grad_cam import get_grad_cam, get_grad_cam_plusplus
from utils.common import time_counter

import models.wrapper
import models.efficientnet.tfkeras
import models.wrapper_T

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
n_classes = 3
batch_size = 64
n_epochs = 100
# lr = 0.001

# Dataset
print('[Dataset]')
with time_counter():
    # with open('train_size128.pkl', 'rb') as fp:
    #     x_train, y_train = pickle.load(fp)
    with open('test_size128.pkl', 'rb') as fp:
        x_test, y_test = pickle.load(fp)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_test.shape[1:]
del x_test, y_test

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
def encoder_net(encoder_model: tfk.Model):
    # inputs = tfk.Input((128, 128, 3))
    inputs = encoder_model.inputs
    normalization_layer = UnitNormLayer()

    encoder = encoder_model
    encoder.trainable = True

    # embeddings = encoder(inputs, training=True)
    embeddings = encoder.layers[-1].output
    embeddings = tfk.layers.GlobalAveragePooling2D()(embeddings)
    norm_embeddings = normalization_layer(embeddings)

    encoder_network = tfk.Model(inputs, norm_embeddings)

    return encoder_network

def supervised_model(encoder_r: tfk.Model, output_size):
    # inputs = tfk.Input((128, 128, 3))
    inputs = encoder_r.inputs
    # encoder_r.trainable = False

    # r = encoder_r(inputs, training=False)
    r = encoder_r.layers[-1].output
    outputs = tfk.layers.Dense(output_size, activation='softmax')(r)

    supervised_model = tfk.Model(inputs, outputs)

    return supervised_model

model = models.wrapper.DenseNet121(weights=None, classes=n_classes, include_top=False, input_shape=in_shape)
enc = encoder_net(model)
linear_model = supervised_model(enc, n_classes)
# o = model.layers[-1].output
# h = tfk.layers.GlobalAveragePooling2D()(o)
# h = UnitNormLayer()(h)
# outputs = tfk.layers.Dense(n_classes, activation='softmax')(h)
# linear_model = tfk.Model(model.inputs, outputs)

weight_name = 'path/to/weights'

linear_model.load_weights(weight_name)

# simclr_model= get_simclr_model(model, 256, 128, 50)
# linear_model = supervised_model(simclr_model, n_classes)

model_name = 'densenet_simclr'

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

save_dir = Path('gcam/')
if not save_dir.exists():
    save_dir.mkdir()

# Validating
# weight_name = 'path to weights'
# weight_name = 'results/validate_acc/best_param_{}'.format(model_name)

# model.load_weights(weight_name)


save_sub_dir = save_dir / (model_name+'/')
if not save_sub_dir.exists():
    save_sub_dir.mkdir()

gcam_list = []
preds = []
trues = []
for label in range(3):
    cnt = 0

    save_sub_sub_dir = save_sub_dir / (str(label)+'/')
    if not save_sub_sub_dir.exists():
        save_sub_sub_dir.mkdir()

    for inputs, labels in test_ds:
        pred = linear_model(inputs, training=False)
        pred = pred.numpy()
        if label == 0:
            true = tf.one_hot(labels, depth=n_classes).numpy()
            preds += [pred]
            trues += [true]
        # h = linear_model.layers[1].layers[1].layers[-4].output
        h = linear_model.layers[-7].output
        # L, pred = get_grad_cam(linear_model, inputs, label, loss, conv_layer=h)
        L, pred = get_grad_cam_plusplus(linear_model, inputs, label, loss, conv_layer=h)
        # L, pred = get_grad_cam(linear_model, inputs, label, loss, final_conv_idx=-10)
        # L, pred = get_grad_cam_plusplus(linear_model, inputs, label, loss, conv_idx=-10)

        pred = np.argmax(pred, axis=1)
        inputs = inputs.numpy()
        labels = labels.numpy()

        width, height = inputs.shape[1:3]
        for org, t, p, gcam in zip(inputs, labels, pred, L):
            gcam = np.uint8(255*gcam)
            resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)
            # resized_gcam = np.transpose(resized_gcam, (1, 0)) # opencvのフォーマットに変換
            org = org * 255.   # opencvのフォーマットに変換
            resized_gcam = cv2.applyColorMap(resized_gcam, cv2.COLORMAP_JET)
            out = cv2.addWeighted(cv2.cvtColor(org.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, resized_gcam, 0.5, 0)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            save_path = save_sub_sub_dir / 'id{}_t{}_p{}.png'.format(cnt, t, p)

            plt.clf()
            plt.imshow(out)
            plt.savefig(str(save_path))
            cnt += 1


preds = np.concatenate(preds)
trues = np.concatenate(trues)
print(np.argmax(preds, axis=1))
acc = np.mean(np.argmax(preds, axis=1) == np.argmax(trues, axis=1))
print(acc)
rows = np.concatenate([trues, preds], axis=1)
df = pd.DataFrame(rows)
df.columns = ['true_normal', 'true_nude', 'true_swimwear', 'pred_normal', 'pred_nude', 'pred_swimwear']
df.to_csv('{}.csv'.format(model_name))

