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
from sklearn.metrics import confusion_matrix

from utils.datasets import load_data
from utils.grad_cam import get_grad_cam, get_grad_cam_plusplus
from utils.common import time_counter

import models.wrapper
import models.efficientnet.tfkeras
import models.wrapper_T

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
n_classes = 2
batch_size = 64
n_epochs = 100
# lr = 0.001

# Dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/sample/', classes=['normal', 'nude', 'swimwear'], size=128, auto_pad_val=True)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

print(_)
print('x: {}'.format(x.shape))
print('y: {}'.format(y.shape))

in_shape = x.shape[1:]
del x, y

model = models.wrapper.DenseNet121(weights=None, classes=n_classes, input_shape=in_shape)
model_name = 'densenet121'

# Validating
accs = {}
print(model_name)
tockens = model_name.split('_')
if len(tockens) == 1:
    train_type = 'normal'
elif len(tockens) == 2:
    train_type = tockens[1]

# weight_name = 'path to weights'

model.load_weights(weight_name)

loss = tfk.losses.SparseCategoricalCrossentropy()

preds = []
trues = []
gcam = []
cnt = 0
for label in range(2):
    for inputs, labels in ds:
        # pred = model(inputs, training=False)
        L, pred = get_grad_cam_plusplus(model, inputs, label, loss, -6)
        # pred = pred.numpy()
        labels = tf.one_hot(labels, depth=n_classes)
        labels = labels.numpy()
        inputs = inputs.numpy()

        preds += [pred]
        trues += [labels]

        pred = np.argmax(pred, axis=1)
        labels = np.argmax(labels, axis=1)

        width, height = inputs.shape[1:3]
        for org, t, p, gcam in zip(inputs, labels, pred, L):
            gcam = np.uint8(255*gcam)
            resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)
            # resized_gcam = np.transpose(resized_gcam, (1, 0)) # opencvのフォーマットに変換
            org = org * 255.   # opencvのフォーマットに変換
            resized_gcam = cv2.applyColorMap(resized_gcam, cv2.COLORMAP_JET)
            out = cv2.addWeighted(cv2.cvtColor(org.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, resized_gcam, 0.5, 0)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            save_path = Path('id{}_t{}_p{}_l{}.png'.format(cnt, t, p, label))

            plt.clf()
            plt.imshow(out)
            plt.savefig(str(save_path))
            cnt += 1

preds = np.concatenate(preds)
trues = np.concatenate(trues)

acc = np.mean(np.argmax(preds, axis=1) == np.argmax(trues, axis=1))
accs[model_name] = acc

# cm = confusion_matrix(np.argmax(trues, axis=1), np.argmax(preds, axis=1))
# df = pd.DataFrame(cm)
# df.columns = ['Normal', 'Nude', 'Swimwear']
# df.index = ['Normal', 'Nude', 'Swimwear']
# df.to_csv('{}_cm.csv'.format(model_name))

rows = np.concatenate([trues, preds], axis=1)
df = pd.DataFrame(rows)
df.columns = ['true_normal', 'true_nude', 'pred_normal', 'pred_nude']
df.to_csv('{}.csv'.format(model_name))

import json
with open('accs.json', 'w') as fp:
    json.dump(accs, fp)