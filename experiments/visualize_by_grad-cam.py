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

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

save_dir = Path('gcam/')
if not save_dir.exists():
    save_dir.mkdir()

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

    # imagenet転移モデルはfunctional apiを使っていないためそのままだとgrad-cam内でエラーが出る
    # 下記のコードを入れて強引にこれを回避
    if train_type == 'pretrained' or train_type == 'ft':
        x = model.layers[0].output
        for layer in model.layers[1].layers:
            x = layer(x)
        model = tfk.Model(model.layers[0].inputs, [x])

    save_sub_dir = save_dir / (model_name+'/')
    if not save_sub_dir.exists():
        save_sub_dir.mkdir()

    gcam_list = []
    for label in range(3):
        cnt = 0

        save_sub_sub_dir = save_sub_dir / (str(label)+'/')
        if not save_sub_sub_dir.exists():
            save_sub_sub_dir.mkdir()

        for inputs, labels in test_ds:
            if train_type == 'normal':
                L, pred = get_grad_cam(model, inputs, label, loss, final_conv_idx)
                # L, pred = get_grad_cam_plusplus(model, inputs, label, loss, final_conv_idx)
            elif train_type == 'pretrained' or train_type == 'ft':
                h = model.get_layer(index=final_conv_idx).output
                L, pred = get_grad_cam(model, inputs, label, loss, conv_layer=h)
                # L, pred = get_grad_cam_plusplus(model, inputs, label, loss, conv_layer=h)
            else:
                raise ValueError('Unknow train type')
            

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


