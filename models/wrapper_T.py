# 転移学習・FT 

import tensorflow as tf
import models

tfk = tf.keras

__all__ = [
    'VGG16', 'VGG19',
    'InceptionV3', 'Xception',
    'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'InceptionResNetV2',
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
    'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'
]

def _create_transfer_model(base, classifier):
    return tfk.Sequential([base, classifier])

# VGG family

def VGG16(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.VGG16(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.Flatten(),
            tfk.layers.Dense(4096, activation='relu', name='fc1'),
            tfk.layers.Dense(4096, activation='relu', name='fc2'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def VGG19(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.VGG19(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.Flatten(),
            tfk.layers.Dense(4096, activation='relu', name='fc1'),
            tfk.layers.Dense(4096, activation='relu', name='fc2'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)


# GoogleNet family

def InceptionV3(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.InceptionV3(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)
 
def Xception(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.Xception(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)


# ResNet family

def ResNet50(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet50(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)
 
def ResNet101(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet101(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)
 
def ResNet152(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet152(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)
 
def ResNet50V2(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet50V2(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def ResNet101V2(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet101V2(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def ResNet152V2(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.ResNet152V2(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def DenseNet121(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.DenseNet121(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def DenseNet169(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.DenseNet169(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

def DenseNet201(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.DenseNet201(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)


# EfficientNet

def EfficientNetBx(model_arch, input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    base = model_arch(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape, **kwargs)
    if 'dropout_rate' not in kwargs:
        dropout_rate = 0.2
    else:
        dropout_rate = kwargs['dropout_rate']
    if classifier is None:
        classifier = tfk.Sequential()
        classifier.add(tfk.layers.GlobalAveragePooling2D(name='avg_pool'))
        if dropout_rate and dropout_rate > 0:
            classifier.add(tfk.layers.Dropout(dropout_rate, name='top_dropout'))
        classifier.add(tfk.layers.Dense(classes,
                       activation='softmax',
                       kernel_initializer=models.efficientnet.model.DENSE_KERNEL_INITIALIZER,
                       name='probs'))
    return _create_transfer_model(base, classifier)

def EfficientNetB0(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB0,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB1(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB1,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB2(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB2,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )
def EfficientNetB3(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB3,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB4(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB4,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB5(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB5,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB6(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB6,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )

def EfficientNetB7(input_shape, classes=3, classifier_activation='softmax', classifier=None, **kwargs):
    return EfficientNetBx(
        models.efficientnet.tfkeras.EfficientNetB7,
        input_shape, classes, classifier_activation, classifier, **kwargs
    )
  
 
# Mixed

def InceptionResNetV2(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.InceptionResNetV2(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

