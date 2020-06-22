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


def training_simCRL(encoder_model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, n_classes, output_best_weights=True, weight_name=None, train_weights=None, **kwargs):
    """
    https://github.com/sayakpaul/SimCLR-in-TensorFlow-2 からのコピペ

    encoder_model: 

    **kwargs
        encoder_opt: 
        encoder_epochs:
    """

    encoder_opt = kwargs.get("encoder_opt", None)
    encoder_epochs = kwargs.get("encoder_epochs", 10)

    def training_SimCLR_Encoder(encoder_model=None, train_ds=None, optimizer=None, n_epochs=20):
        """
        自己教師ありの部分
        """
        
        if train_ds is None:
            raise ValueError("train_ds most not be None")

        if encoder_model is None:
            # encoder_model = ResNet50(weights=None, include_top=False)
            raise ValueError("encoder_model most not be None")

        if optimizer is None:
            optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
            
        # Image Augmentation
        class CustomAugment(object):
            # Augmentation utilities (differs from the original implementation)
            # Referred from: https://arxiv.org/pdf/2002.05709.pdf (Appendxi A 
            # corresponding GitHub: https://github.com/google-research/simclr/)
            def __call__(self, sample):
                # Random flips
                sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
                
                # Randomly apply transformation (color distortions) with probability p.
                sample = self._random_apply(self._color_jitter, sample, p=0.8)
                sample = self._random_apply(self._color_drop, sample, p=0.2)

                return sample

            def _color_jitter(self, x, s=1):
                # one can also shuffle the order of following augmentations
                # each time they are applied.
                x = tf.image.random_brightness(x, max_delta=0.8*s)
                x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
                x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
                x = tf.image.random_hue(x, max_delta=0.2*s)
                x = tf.clip_by_value(x, 0, 1)
                return x
            
            def _color_drop(self, x):
                x = tf.image.rgb_to_grayscale(x)
                x = tf.tile(x, [1, 1, 1, 3])
                return x
            
            def _random_apply(self, func, x, p):
                return tf.cond(
                tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                        tf.cast(p, tf.float32)),
                lambda: func(x),
                lambda: x)

        # Build the augmentation pipeline
        data_augmentation = tfk.Sequential([tfk.layers.Lambda(CustomAugment())])
        # # TODO 画像拡張の確認
        # import matplotlib.pyplot as plt
        # images, labels = next(iter(train_ds))
        # aug_images = data_augmentation(images)
        # for img_idx in range(5):
        #     print(labels[img_idx])
        #     ax1 = plt.subplot(1, 2, 1)
        #     ax2 = plt.subplot(1, 2, 2)
        #     ax1.imshow(images[img_idx])
        #     ax2.imshow(aug_images[img_idx])
        #     plt.show()
        # sys.exit()

        # Architecture utils
        def get_simclr_model(base: tfk.Model, hidden_1=256, hidden_2=128, hidden_3=50):
            inputs = tfk.Input((128, 128, 3))

            base_model = base
            base_model.trainabe = True

            h = base_model(inputs, training=True)
            h = tfk.layers.GlobalAveragePooling2D()(h)

            projection_1 = tfk.layers.Dense(hidden_1)(h)
            projection_1 = tfk.layers.Activation("relu")(projection_1)
            projection_2 = tfk.layers.Dense(hidden_2)(projection_1)
            projection_2 = tfk.layers.Activation("relu")(projection_2)
            projection_3 = tfk.layers.Dense(hidden_3)(projection_2)

            resnet_simclr = tfk.Model(inputs, projection_3)

            return resnet_simclr


        def get_negative_mask(batch_size):
            # return a mask that removes the similarity score of equal/similar images.
            # this function ensures that only distinct pair of images get their similarity scores
            # passed as negative examples
            negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
            for i in range(batch_size):
                negative_mask[i, i] = 0
                negative_mask[i, i + batch_size] = 0
            return tf.constant(negative_mask)
        # Mask to remove positive examples from the batch of negative samples

        from utils.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
        @tf.function
        def train_step(xis, xjs, model, optimizer, criterion, temperature):
            BATCH_SIZE = xis.shape[0]
            with tf.GradientTape() as tape:
                zis = model(xis)
                zjs = model(xjs)

                # normalize projection feature vectors
                zis = tf.math.l2_normalize(zis, axis=1)
                zjs = tf.math.l2_normalize(zjs, axis=1)

                l_pos = sim_func_dim1(zis, zjs)
                l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
                l_pos /= temperature

                negatives = tf.concat([zjs, zis], axis=0)

                loss = 0

                for positives in [zis, zjs]:
                    l_neg = sim_func_dim2(positives, negatives)

                    labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

                    negative_mask = get_negative_mask(BATCH_SIZE)

                    l_neg = tf.boolean_mask(l_neg, negative_mask)
                    l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
                    l_neg /= temperature

                    logits = tf.concat([l_pos, l_neg], axis=1) 
                    loss += criterion(y_pred=logits, y_true=labels)

                loss = loss / (2 * BATCH_SIZE)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss

        def train_simclr(model, dataset, optimizer, criterion, temperature=0.1, epochs=100):
            epoch_wise_loss = []

            epoch_loss_avg = tfk.metrics.Mean()
            for epoch in range(epochs):
                for image_batch, _ in dataset:
                    a = data_augmentation(image_batch)
                    b = data_augmentation(image_batch)

                    loss = train_step(a, b, model, optimizer, criterion, temperature)
                    epoch_loss_avg.update_state(loss)
                
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, epoch_loss_avg.result()))

                epoch_wise_loss.append(epoch_loss_avg.result().numpy())

                epoch_loss_avg.reset_states()

            return epoch_wise_loss, model

        # Loss
        loss = tfk.losses.SparseCategoricalCrossentropy()

        # model
        simclr_model = get_simclr_model(encoder_model, 256, 128, 50)

        # Training
        epoch_wise_loss, simclr_model  = train_simclr(simclr_model, train_ds, optimizer, loss, temperature=0.1, epochs=n_epochs)

        hist = {"'nt_xentloss'": epoch_wise_loss}
        return simclr_model, hist

    simclr_model, hist_nt_xentloss = training_SimCLR_Encoder(encoder_model=encoder_model, optimizer=encoder_opt, train_ds=train_ds, n_epochs=encoder_epochs)

    # =============================================================================
    # Fine-tuning
    # =============================================================================

    # model
    def supervised_model(projection: tfk.Model, output_size):
        inputs = tfk.Input((128, 128, 3))
        projection.trainable = False

        r = projection(inputs, training=False)
        outputs = tfk.layers.Dense(output_size, activation='softmax')(r)

        supervised_model = tfk.Model(inputs, outputs)
    
        return supervised_model

    # Encoder model with non-linear projections
    projection = tfk.Model(simclr_model.input, simclr_model.layers[-2].output)
    linear_model = supervised_model(projection, n_classes)

    # Training
    # TODO train_ds.take(5):の部分　訓練データの量をどうするか
    hist = training(linear_model, train_ds.take(5), test_ds, loss, optimizer, n_epochs, batch_size, weight_name=weight_name)

    return hist

