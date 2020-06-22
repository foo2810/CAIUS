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
        
        if output_best_weights and best_acc < test_acc.result().numpy():
            print("test_acc is improved {.3f} to {.3f}".format(best_acc, test_acc.result().numpy()))
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
        
        if output_best_weights and best_acc < test_acc.result().numpy():
            print("test_acc is improved {.3f} to {.3f}".format(best_acc, test_acc.result().numpy()))
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

