import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import optuna
import numpy as np

from utils.train import training_mixup
from utils.data_augment import random_rotate_90, random_flip_left_right, gen_random_cutout
from utils.datasets import load_data, train_test_split
from utils.common import time_counter

from models.wrapper_T import *

import tensorflow as tf
tfk = tf.keras


# Hyper Parameters
n_classes = 3
batch_size = 64
n_epochs = 50

# dataset
print('[Dataset]')
with time_counter():
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128_autopad.pkl', auto_pad_val=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)
    n_train = len(x_train)

    _random_cutout = gen_random_cutout(42)
    @tf.function
    def augment(image, label):
        image, label = random_flip_left_right(image, label)
        # image, label = random_flip_up_down(image, label)
        image, label =_random_cutout(image, label)
        image, label = random_rotate_90(image, label)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(n_train).map(augment) \
        .batch(batch_size).repeat(3)    # datasetのサイズを3倍に
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_train.shape[1:]
del x_train, y_train, x_test, y_test

def get_optimizer(trial):
    lr = trial.suggest_loguniform('adam_lr', 1e-6, 1e-4)
    optimizer =  tf.optimizers.Adam(lr)
    return optimizer

def get_alpha(trial):
    alpha = trial.suggest_uniform('alpha_mixup', 0.1, 0.4)
    return alpha

def objective(trial):
    model = VGG16(classes=n_classes, input_shape=in_shape)
    loss = tfk.losses.CategoricalCrossentropy()
    opt = get_optimizer(trial)
    alpha = get_alpha(trial)

    train_weights = model.layers[1].trainable_weights

    hist = training_mixup(model, train_ds, test_ds, loss, opt, n_epochs, batch_size, n_classes, alpha, output_best_weights=False, train_weights=train_weights)

    test_acc = np.array(hist['test_acc'])

    return 1 - test_acc.max()


trial_size = 5
study = optuna.create_study()
study.optimize(objective, n_trials=trial_size)

print('[Best params]')
print(study.best_params)

print('[Best values (error rate)]')
print(study.best_value)

print('[Search process]')
fig = optuna.visualization.plot_optimization_history(study)
img = fig.to_image('jpg')
with open('search_process.jpg', 'wb') as fp:
    fp.write(img)

fig2 = optuna.visualization.plot_intermediate_values(study)
img2 = fig.to_image('jpg')
with open('intermediate_values.jpg', 'wb') as fp:
    fp.write(img2)

df = study.trials_dataframe()
df = df.to_csv('history_of_searching_hparams.csv')
