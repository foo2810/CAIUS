import tensorflow as tf
import numpy as np

tfk = tf.keras

def training(model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, output_best_weights=True, weight_name=None, train_weights=None):
    ## Metrics
    train_loss = tf.metrics.Mean(name='train_loss')
    test_loss = tf.metrics.Mean(name='test_loss')
    train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_acc = tf.metrics.SparseCategoricalAccuracy(name='test_acc')

    ## Loss
    # loss = tfk.losses.SparseCategoricalCrossentropy()

    ## Optimizer
    # opt = tfk.optimizers.Adam(lr=lr)

    if train_weights is None:
        train_weights = model.trainable_variables

    @tf.function
    def train_step(model, inputs, labels):
        with tf.GradientTape() as tape:
            pred = model(inputs)
            loss_val = loss(labels, pred)
        # grads = tape.gradient(loss_val, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        grads = tape.gradient(loss_val, train_weights)
        optimizer.apply_gradients(zip(grads, train_weights))

        train_loss(loss_val)
        train_acc(labels, pred)

    @tf.function
    def test_step(model, inputs, labels):
        pred = model(inputs)
        loss_val = loss(labels, pred)
        test_loss(loss_val)
        test_acc(labels, pred)

    hist = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    template = 'Epoch[{}/{}] loss: {:.3f}, acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}'
    best_acc = 0
    for epoch in range(n_epochs):
        for inputs, labels in train_ds:
            train_step(model, inputs, labels)
        
        for inputs, labels in test_ds:
            test_step(model, inputs, labels)
        
        if output_best_weights and  best_acc < test_acc.result().numpy():
            best_acc = test_acc.result().numpy()
            if weight_name is None:
                weight_name = 'best_param'
            model.save_weights(weight_name, save_format='tf')
        
        print(template.format(
            epoch+1, n_epochs,
            train_loss.result().numpy(),
            train_acc.result().numpy(),
            test_loss.result().numpy(),
            test_acc.result().numpy(),
        ))

        hist['train_loss'] += [train_loss.result().numpy()]
        hist['test_loss'] += [test_loss.result().numpy()]
        hist['train_acc'] += [train_acc.result().numpy()]
        hist['test_acc'] += [test_acc.result().numpy()]
        
        train_loss.reset_states()
        test_loss.reset_states()
        train_acc.reset_states()
        test_acc.reset_states()
    
    return hist

def _mixup(inputs, onehot_labels, alpha):
    inputs_rv = tf.reverse(inputs, axis=[0])
    # onehot_labels = tf.one_hot(labels, n_classes)
    onehot_labels_rv = tf.reverse(onehot_labels, axis=[0])
    l = tf.constant(np.random.beta(alpha, alpha, size=(inputs.shape[0], 1, 1, 1)))

    mixed_inputs = l * inputs + (1 - l) * inputs_rv
    mixed_labels = tf.cast(l[:, :, 0, 0], onehot_labels.dtype) * onehot_labels + tf.cast(1 - l[:, :, 0, 0], onehot_labels.dtype) * onehot_labels_rv

    return mixed_inputs, mixed_labels


# loss: Sparse...は指定できない
def training_mixup(model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, n_classes, alpha, output_best_weights=True, weight_name=None, train_weights=None):
    ## Metrics
    train_loss = tf.metrics.Mean(name='train_loss')
    test_loss = tf.metrics.Mean(name='test_loss')
    train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_acc = tf.metrics.SparseCategoricalAccuracy(name='test_acc')

    ## Loss
    # loss = tfk.losses.SparseCategoricalCrossentropy()

    ## Optimizer
    # opt = tfk.optimizers.Adam(lr=lr)

    if train_weights is None:
        train_weights = model.trainable_variables
    
    @tf.function
    def train_step(model, inputs, labels):
        with tf.GradientTape() as tape:
            pred = model(inputs)
            loss_val = loss(tf.one_hot(labels, n_classes), pred)
        # grads = tape.gradient(loss_val, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        grads = tape.gradient(loss_val, train_weights)
        optimizer.apply_gradients(zip(grads, train_weights))
        del tape

        mixed_inputs, mixed_labels = _mixup(inputs, tf.one_hot(labels, n_classes), alpha)
        with tf.GradientTape() as tape:
            pred = model(mixed_inputs)
            loss_val = loss(mixed_labels, pred)
        # grads = tape.gradient(loss_val, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        grads = tape.gradient(loss_val, train_weights)
        optimizer.apply_gradients(zip(grads, train_weights))

        train_loss(loss_val)
        train_acc(labels, pred)

    @tf.function
    def test_step(model, inputs, labels):
        pred = model(inputs)
        loss_val = loss(tf.one_hot(labels, n_classes), pred)
        test_loss(loss_val)
        test_acc(labels, pred)

    hist = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    template = 'Epoch[{}/{}] loss: {:.3f}, acc: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}'
    best_acc = 0
    for epoch in range(n_epochs):
        for inputs, labels in train_ds:
            train_step(model, inputs, labels)
        
        for inputs, labels in test_ds:
            test_step(model, inputs, labels)
        
        if output_best_weights and  best_acc < test_acc.result().numpy():
            best_acc = test_acc.result().numpy()
            if weight_name is None:
                weight_name = 'best_param'
            model.save_weights(weight_name, save_format='tf')
        
        print(template.format(
            epoch+1, n_epochs,
            train_loss.result().numpy(),
            train_acc.result().numpy(),
            test_loss.result().numpy(),
            test_acc.result().numpy(),
        ))

        hist['train_loss'] += [train_loss.result().numpy()]
        hist['test_loss'] += [test_loss.result().numpy()]
        hist['train_acc'] += [train_acc.result().numpy()]
        hist['test_acc'] += [test_acc.result().numpy()]
        
        train_loss.reset_states()
        test_loss.reset_states()
        train_acc.reset_states()
        test_acc.reset_states()
    
    return hist


from utils.losses import SupConLoss
def training_supCon(encoder_model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, n_classes, output_best_weights=True, weight_name=None, train_weights=None, **kwargs):
    """
    https://github.com/sayakpaul/Supervised-Contrastive-Learning-in-TensorFlow-2 からのコピペ

    encoder_model: 

    **kwargs
        encoder_opt: 
        encoder_epochs:
    """

    encoder_opt = kwargs.get("encoder_opt", None)
    encoder_epochs = kwargs.get("encoder_epochs", 10)

    def training_SupCon_Encoder(encoder_model=None, train_ds=None, optimizer=None, n_epochs=20):
        """
        Encoder nework and Projection network 自己教師ありの部分
        """

        if train_ds is None:
            raise ValueError("train_ds most not be None")

        if encoder_model is None:
            # encoder_model = ResNet50(weights=None, include_top=False)
            raise ValueError("encoder_model most not be None")

        if optimizer is None:
            optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
            

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
            inputs = tfk.Input((128, 128, 3))
            normalization_layer = UnitNormLayer()

            encoder = encoder_model
            encoder.trainable = True

            embeddings = encoder(inputs, training=True)
            embeddings = tfk.layers.GlobalAveragePooling2D()(embeddings)
            norm_embeddings = normalization_layer(embeddings)

            encoder_network = tfk.Model(inputs, norm_embeddings)

            return encoder_network

        # Projector Network
        def projector_net(encoder_r: tfk.Model):
            encoder_r.trainable = True
            projector = tfk.models.Sequential([
                encoder_r,
                tfk.layers.Dense(256, activation="relu"),
                UnitNormLayer()
            ])

            return projector

        # Training the encoder and the projector

        encoder_r = encoder_net(encoder_model)
        projector_z = projector_net(encoder_r)
        supConloss = SupConLoss()

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                z = projector_z(images, training=True)
                loss = supConloss(z, labels)

            gradients = tape.gradient(loss, projector_z.trainable_variables)
            optimizer.apply_gradients(zip(gradients, projector_z.trainable_variables))

            return loss

        train_loss_results = []

        epoch_loss_avg = tf.keras.metrics.Mean()
        for epoch in range(n_epochs):	
            for (images, labels) in train_ds:
                loss = train_step(images, labels)
                epoch_loss_avg.update_state(loss) 

                # TODO loss が nan になることがある
                if tf.math.is_nan(loss):
                    print("loss", loss)
                    print("encoder_r output is nan:", tf.reduce_any(tf.math.is_nan(encoder_r(images))).numpy())
                    print("projector_z output is nan:", tf.reduce_any(tf.math.is_nan(projector_z(images))).numpy())
                    import sys;sys.exit()

            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, n_epochs, epoch_loss_avg.result()))

            train_loss_results.append(epoch_loss_avg.result().numpy())

            epoch_loss_avg.reset_states()

        hist = {"'supervised_contrastive_loss'": train_loss_results}
        return encoder_r, hist

    # Training for Encoder Network
    encoder_r, hist_cupCon = training_SupCon_Encoder(encoder_model=encoder_model, optimizer=encoder_opt, train_ds=train_ds, n_epochs=encoder_epochs)

    # =============================================================================
    # Classifer 部分　教師あり学習
    # =============================================================================

    # model
    def supervised_model(encoder_r: tfk.Model, output_size):
        inputs = tfk.Input((128, 128, 3))
        encoder_r.trainable = False

        r = encoder_r(inputs, training=False)
        outputs = tfk.layers.Dense(output_size, activation='softmax')(r)

        supervised_model = tfk.Model(inputs, outputs)
    
        return supervised_model

    model = supervised_model(encoder_r, n_classes)

    # Training
    hist = training(model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, weight_name=weight_name)

    return hist

