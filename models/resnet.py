import tensorflow as tf

tfk = tf.keras

def _conv_bn_relu(nb_fil, nb_row, strides):
    def f(input):
        conv = tfk.layers.Conv2D(nb_fil, nb_row, strides=strides, kernel_initializer="he_normal", padding="same")(input)
        norm = tfk.layers.BatchNormalization()(conv)
        return tfk.layers.ReLU()(norm)
    return f

def _bn_relu_conv(nb_fil, nb_row, strides=1, padding="same"):
    def f(input):
        norm = tfk.layers.BatchNormalization()(input)
        act = tfk.layers.ReLU()(norm)
        return tfk.layers.Conv2D(nb_fil, nb_row, strides=strides, kernel_initializer="he_normal", padding=padding)(act)
    return f

def _basic_block(nb_fil, strides):
    def f(input):
        conv1 = _bn_relu_conv(nb_fil, 3, strides=strides)(input)
        residual = _bn_relu_conv(nb_fil, 3)(conv1)
        return _shortcut(input, residual, stride=strides)
    return f

def _bottleneck_block(nb_fil, strides):
    print('strides:', strides)
    def f(input):
        conv1 = _bn_relu_conv(nb_fil, 1, strides=1)(input)
        conv2 = _bn_relu_conv(nb_fil, 3, strides=strides, padding="same")(conv1)
        conv3 = _bn_relu_conv(nb_fil*4, 1, strides=1)(conv2)
        return _shortcut(input, conv3, stride=strides)
    return f
        

def _shortcut(input, residual, stride=None):
    # misstake
    #stride = input._keras_shape[2]/residual._keras_shape[2]
    #equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    # stride = input._keras_shape[1] / residual._keras_shape[1]
    # equal_channels = residual._keras_shape[2] == input._keras_shape[2]

    # for tensorflow.keras
    equal_channels = residual.shape[2] == input.shape[2]
    if stride is None:
        stride = input.shape[1] // residual.shape[1]
    
    shortcut = input
    if stride > 1 or not equal_channels:
        #shortcut = Conv1D(residual._keras_shape[1], 1, strides=int(stride), kernel_initializer='he_normal', padding='valid')(input)

        #これが先生からもらった時のコード
        #shortcut = Conv1D(residual._keras_shape[2], 1, strides=int(stride), kernel_initializer='he_normal', padding='valid')(input)
        shortcut = tfk.layers.Conv2D(residual.shape[2], 1, strides=int(stride), kernel_initializer='he_normal', padding='valid')(input)

    #return merge([shortcut, residual], mode="sum")
    return tfk.layers.add([shortcut, residual])

def _residual_block(block_function, nb_fil, repetations, is_first_layer = False):
    def f(input):
        for i in range(repetations):
            init_stride = 1
            if i == 0 and not is_first_layer:
                init_stride = 2

            input = block_function(nb_fil, strides=init_stride)(input)
        return input
    return f

def resnet34(inshape, n_classes):
    in_tensor = tfk.Input(shape=inshape)
    conv1 = _conv_bn_relu(nb_fil = 64, nb_row=7, strides=2)(in_tensor)
    pool1 = tfk.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv1)

    # Plain arch
    block_fn = _basic_block
    block1 = _residual_block(block_fn, nb_fil = 64, repetations=3, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_fil = 128, repetations=4)(block1)
    block3 = _residual_block(block_fn, nb_fil = 256, repetations=4)(block2)
    block4 = _residual_block(block_fn, nb_fil = 512, repetations=4)(block3)
    block5 = _residual_block(block_fn, nb_fil = 1024, repetations=4)(block4)
    blk = block5

    # Bottleneck arch (ResNet50 ...)
    # block_fn = _bottleneck_block
    # block1 = _residual_block(block_fn, nb_fil = 64, repetations=3, is_first_layer=True)(pool1)
    # block2 = _residual_block(block_fn, nb_fil = 128, repetations=4)(block1)
    # block3 = _residual_block(block_fn, nb_fil = 256, repetations=4)(block2)
    # block4 = _residual_block(block_fn, nb_fil = 512, repetations=4)(block3)
    # blk = block4
    
    pool2 = tfk.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(blk)
    flat = tfk.layers.Flatten()(pool2)
    dense = tfk.layers.Dense(n_classes, kernel_initializer='he_normal')(flat)
    out = tfk.layers.Softmax(dense)
    
    model = tfk.Model(inputs=in_tensor, outputs=out)
    return model

