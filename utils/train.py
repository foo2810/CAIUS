import tensorflow as tf

tfk = tf.keras

def training(model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, output_best_weights=True, train_weights=None):
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
            model.save_weights('best_param', save_format='tf')
        
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

