# 転移学習・FT 

import tensorflow as tf

tfk = tf.keras

__all__ = [
    'VGG16', 'VGG19',
    'InceptionV3', 'Xception',
    'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'InceptionResNetV2',
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

def EfficientNetB0(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    ...

# Mixed

def InceptionResNetV2(input_shape, classes=3, classifier_activation='softmax', classifier=None):
    base = tfk.applications.InceptionResNetV2(include_top=False, weights='imagenet', classes=classes, input_shape=input_shape)
    if classifier is None:
        classifier = tfk.Sequential([
            tfk.layers.GlobalAveragePooling2D(name='avg_pool'),
            tfk.layers.Dense(classes, activation=classifier_activation, name='predictions'),
        ])
    return _create_transfer_model(base, classifier)

