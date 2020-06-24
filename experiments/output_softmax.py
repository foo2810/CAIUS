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

models_and_params = {
    'vgg16': (models.wrapper.VGG16(weights=None, classes=3, input_shape=in_shape), -6),
    'resnet50': (models.wrapper.ResNet50(weights=None, classes=3, input_shape=in_shape), -6),
    'inceptionv3': (models.wrapper.InceptionV3(weights=None, classes=3, input_shape=in_shape), -14),
    'densenet121': (models.wrapper.DenseNet121(weights=None, classes=3, input_shape=in_shape), -6),
    'inceptionresnetv2': (models.wrapper.InceptionResNetV2(weights=None, classes=3, input_shape=in_shape), -5),
    'efficientnetb0': (models.efficientnet.tfkeras.EfficientNetB0(weights=None, classes=3, input_shape=in_shape), -6),

    'vgg16_pretrained': (models.wrapper_T.VGG16(classes=3, input_shape=in_shape), -6),
    'resnet50_pretrained': (models.wrapper_T.ResNet50(classes=3, input_shape=in_shape), -6),
    'inceptionv3_pretrained': (models.wrapper_T.InceptionV3(classes=3, input_shape=in_shape), -3),
    'densenet121_pretrained': (models.wrapper_T.DenseNet121( classes=3, input_shape=in_shape), -6),
    'inceptionresnetv2_pretrained': (models.wrapper_T.InceptionResNetV2(classes=3, input_shape=in_shape), -5),
    'efficientnetb0_pretrained': (models.wrapper_T.EfficientNetB0(classes=3, input_shape=in_shape), -6),

    'vgg16_ft': (models.wrapper_T.VGG16(classes=3, input_shape=in_shape), -6),
    'resnet50_ft': (models.wrapper_T.ResNet50(classes=3, input_shape=in_shape), -6),
    'inceptionv3_ft': (models.wrapper_T.InceptionV3(classes=3, input_shape=in_shape), -3),
    'densenet121_ft': (models.wrapper_T.DenseNet121( classes=3, input_shape=in_shape), -6),
    'inceptionresnetv2_ft': (models.wrapper_T.InceptionResNetV2(classes=3, input_shape=in_shape), -5),
    'efficientnetb0_ft': (models.wrapper_T.EfficientNetB0(classes=3, input_shape=in_shape), -6),
}

# Validating
for model_name in models_and_params:
    print(model_name)
    tockens = model_name.split('_')
    if len(tockens) == 1:
        train_type = 'normal'
    elif len(tockens) == 2:
        train_type = tockens[1]

    model, final_conv_idx = models_and_params[model_name]
    # weight_name = 'path to weights'
    weight_name = 'results/validate_acc/best_param_{}'.format(model_name)

    model.load_weights(weight_name)

    preds = []
    trues = []
    for inputs, labels in test_ds:
        pred = model(inputs)
        pred = pred.numpy()
        labels = tf.one_hot(labels, depth=n_classes)
        labels = labels.numpy()

        preds += [pred]
        trues += [labels]
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    rows = np.concatenate([trues, preds], axis=1)
    df = pd.DataFrame(rows)
    df.columns = ['true_normal', 'true_nude', 'true_swimwear', 'pred_normal', 'pred_nude', 'pred_swimwear']
    df.to_csv('{}.csv'.format(model_name))
