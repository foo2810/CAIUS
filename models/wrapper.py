import tensorflow as tf
from models.efficientnet.tfkeras import *

tfk = tf.keras

# VGG family

def VGG16(**kwargs):
    return tfk.applications.VGG16(**kwargs)

def VGG19(**kwargs):
    return tfk.applications.VGG19(**kwargs)

# GoogleNet family

def InceptionV3(**kwargs):
    return tfk.applications.InceptionV3(**kwargs)

def Xception(**kwargs):
    return tfk.applications.Xception(**kwargs)

# ResNet family

def ResNet50(**kwargs):
    return tfk.applications.ResNet50(**kwargs)

def ResNet101(**kwargs):
    return tfk.applications.ResNet101(**kwargs)

def ResNet152(**kwargs):
    return tfk.applications.ResNet152(**kwargs)

def ResNet50V2(**kwargs):
    return tfk.applications.ResNet50V2(**kwargs)

def ResNet101V2(**kwargs):
    return tfk.applications.ResNet101V2(**kwargs)

def ResNet152V2(**kwargs):
    return tfk.applications.ResNet152V2(**kwargs)

def DenseNet121(**kwargs):
    return tfk.applications.DenseNet121(**kwargs)

def DenseNet169(**kwargs):
    return tfk.applications.DenseNet121(**kwargs)

def DenseNet201(**kwargs):
    return tfk.applications.DenseNet121(**kwargs)

# EfficientNet
# models.efficientnet.tfkerasのものをそのまま使う

# Mixed

def InceptionResNetV2(**kwargs):
    return tfk.applications.InceptionResNetV2(**kwargs)

