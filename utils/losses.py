import tensorflow as tf

tfk = tf.keras

class ComplementEntropy(tfk.losses.Loss):
    def call(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        n_classes = y_pred.shape[1]
        # y_true = tf.cast(y_true, y_pred.dtype)
        # print('y_true: ', y_true.shape)
        # print('y_pred: ', y_pred.shape)

        loss_base = tfk.losses.sparse_categorical_crossentropy(y_true, y_pred)
        # print('loss_base: ', loss_base.shape)

        gt_mask = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), n_classes), tf.bool)
        yhat_gt = tf.boolean_mask(y_pred, gt_mask)
        yhat_gt = tf.reshape(yhat_gt, (batch_size, 1))
        # print('yhat_gt: ', yhat_gt.shape)
        yhat_gt_stable = (1 - yhat_gt) + tf.constant(1e-7, dtype=y_pred.dtype)
        # print('yhat_gt_stable: ', yhat_gt_stable.shape)
        px = y_pred / tf.reshape(yhat_gt_stable, (batch_size, 1))
        # print('px: ', px.shape)
        log_px = tf.math.log(px + tf.constant(1e-10, dtype=y_pred.dtype))
        # print('log_px: ', log_px.shape)
        mask = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        # print('mask: ', mask.shape)
        loss_compl = px * log_px * mask
        loss_compl = tf.reduce_sum(loss_compl) \
            / tf.constant(batch_size, dtype=y_pred.dtype) \
            / tf.constant(n_classes, dtype=y_pred.dtype)
        
        loss = loss_base + loss_compl
        return loss
