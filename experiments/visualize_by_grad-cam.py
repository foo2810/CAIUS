import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from utils.grad_cam import get_grad_cam
from utils.common import time_counter

import models

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
    # with open('train_size128.pkl') as fp:
    #     x_train, y_train = pickle.load(fp)
    with open('test_size128.pkl') as fp:
        x_test, y_test = pickle.load(fp)

    _random_cutout = gen_random_cutout(42)
    @tf.function
    def augment(image, label):
        image, label = random_flip_left_right(image, label)
        # image, label = random_flip_up_down(image, label)
        image, label =_random_cutout(image, label)
        image, label = random_rotate_90(image, label)
        return image, label

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_test.shape[1:]
del x_test, y_test

models_and_params = {
    'resnet50': (models.wrapper.ResNet50(weights=None, classes=3, input_shape=in_shape), -6),
    'inceptionv3': (models.wrapper.InceptionV3(weights=None, classes=3, input_shape=in_shape), ???),
    'densenet121': (models.wrapper.DenseNet121(weights=None, classes=3, input_shape=in_shape), ???),
    'inceptionresnetv2': (models.wrapper.InceptionResNetV2(weights=None, classes=3, input_shape=in_shape), ???),
    # 'efficientnet': (models.efficientnet.tfkeras.EfficientNetB0(weights=None, classes=3, input_shape=in_shape), (,)),

    'resnet50_pretraind': (models.wrapper_T.ResNet50(classes=3, input_shape=in_shape), -6),
    'inceptionv3_pretrained': (models.wrapper_T.InceptionV3(classes=3, input_shape=in_shape), ???),
    'densenet121_pretrained': (models.wrapper_T.DenseNet121( classes=3, input_shape=in_shape), ???),
    'inceptionresnetv2_pretrained': (models.wrapper_T.InceptionResNetV2(classes=3, input_shape=in_shape), ???),
    # 'efficientnet': (models.efficientnet.tfkeras.EfficientNetB0(weights=None, classes=3, input_shape=in_shape), (,)),

    'resnet50_ft': (models.wrapper_T.ResNet50(classes=3, input_shape=in_shape), -6),
    'inceptionv3_ft': (models.wrapper_T.InceptionV3(classes=3, input_shape=in_shape), ???),
    'densenet121_ft': (models.wrapper_T.DenseNet121( classes=3, input_shape=in_shape), ???),
    'inceptionresnetv2_ft': (models.wrapper_T.InceptionResNetV2(classes=3, input_shape=in_shape), ???),
    # 'efficientnet': (models.efficientnet.tfkeras.EfficientNetB0(weights=None, classes=3, input_shape=in_shape), (,)),
}

# Loss
loss = tfk.losses.SparseCategoricalCrossentropy()

save_dir = Path('gcam/')
if not save_dir.exists():
    save_dir.mkdir()

# Validating
for model_name in models_and_params:
    model, final_conv_idx = models_and_params[model_name]
    weight_name = 'best_param_{}'.format(model_name)

    model.load_weights(weight_name)

    save_sub_dir = save_dir / model_name+'/'
    if not save_sub_dir.exists():
        save_sub_dir.mkdir()

    gcam_list = []
    for label in range(3):
        cnt = 0

        save_sub_sub_dir = save_sub_dir / str(label)+'/'
        if not save_sub_sub_dir.exists():
            save_sub_sub_dir.mkdir()

        for inputs, labels in test_ds:
            L, pred = get_grad_cam(model, inputs, label, loss, final_conv_idx)
            pred = np.argmax(pred, axis=1)
            inputs = inputs.numpy()
            labels = labels.numpy()

            width, height = inputs.shape[1:3]
            for org, t, p, gcam in zip(inputs, labels, pred, L):
                gcam = np.uint8(255*gcam)
                resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)
                resized_gcam = np.transpose(resized_gcam, (1, 0)) # opencvのフォーマットに変換
                org = np.transpose(org, (1, 0, 2)) * 255.   # opencvのフォーマットに変換
                resized_gcam = cv2.applyColorMap(resized_gcam, cv2.COLORMAP_JET)
                out = cv2.addWeighted(cv2.cvtColor(org.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, resized_gcam, 0.5, 0)

                save_path = save_sub_sub_dir / 'id{}_t{}_p{}.png'.format(cnt, t, p)

                plt.clf()
                plt.imshow(out)
                plt.savefig(str(save_path))
                cnt += 1


