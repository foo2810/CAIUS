import tensorflow as tf

tfk = tf.keras

class SparseComplementEntropy(tfk.losses.Loss):
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

# Mixupを使うとComplementEntropyが使えない...
class ComplementEntropy(tfk.losses.Loss):
    def call(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        n_classes = y_pred.shape[1]
        # y_true = tf.cast(y_true, y_pred.dtype)
        # print('y_true: ', y_true.shape)
        # print('y_pred: ', y_pred.shape)

        loss_base = tfk.losses.categorical_crossentropy(y_true, y_pred)
        # print('loss_base: ', loss_base.shape)

        gt_mask = tf.cast(tf.cast(y_true, tf.int32), tf.bool)
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



# https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/contrastive.py
from typing import Union, List
import numpy as np
import tensorflow as tf
Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
]
@tf.function
def contrastive_loss(
    y_true: TensorLike, y_pred: TensorLike, margin: Number = 1.0
) -> tf.Tensor:
    r"""Computes the contrastive loss between `y_true` and `y_pred`.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape [batch_size, hidden_size] can be computed
    as follows:
    ```python
    # y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
    y_pred = tf.linalg.norm(a - b, axis=1)
    ```
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        binary labels indicating positive vs negative pair.
      y_pred: 1-D float `Tensor` with shape [batch_size] of
        distances between two embedding matrices.
      margin: margin term in the loss definition.
    Returns:
      contrastive_loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    )


class SupConLoss(tfk.losses.Loss):
    """
    (max_margin_contrastive_loss)  

    参考  
    https://github.com/sayakpaul/Supervised-Contrastive-Learning-in-TensorFlow-2  
    https://github.com/wangz10/contrastive_loss/blob/master/model.py  
    https://github.com/wangz10/contrastive_loss/blob/master/losses.py  
    https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/contrastive.py  
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py  

    """
    def call(self, z, y, margin=1.0, metric='euclidean'):
        '''
        Wrapper for the maximum margin contrastive loss (Hadsell et al. 2006)
        `tfa.losses.contrastive_loss`
        Args:
            z: hidden vector of shape [bsz, n_features].
            y: ground truth of shape [bsz].
            metric: one of ('euclidean', 'cosine')
        '''
        # compute pair-wise distance matrix
        if metric == 'euclidean':
            D = self.__pdist_euclidean(z)
        elif metric == 'cosine':
            D = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
        # convert squareform matrix to vector form
        d_vec = self.__square_to_vec(D)
        # make contrastive labels
        y_contrasts = self.__get_contrast_batch_labels(y)
        loss = contrastive_loss(y_contrasts, d_vec, margin=margin)
        # exploding/varnishing gradients on large batch?
        return tf.reduce_mean(loss)
    
    def __pdist_euclidean(self, A):
        # Euclidean pdist
        # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        r = tf.reduce_sum(A*A, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
        return tf.sqrt(D)

    def __square_to_vec(self, D):
        '''Convert a squared form pdist matrix to vector form.
        '''
        n = D.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        d_vec = tf.gather_nd(D, list(zip(triu_idx[0], triu_idx[1])))
        return d_vec

    def __get_contrast_batch_labels(self, y):
        '''
        Make contrast labels by taking all the pairwise in y
        y: tensor with shape: (batch_size, )
        returns:   
            tensor with shape: (batch_size * (batch_size-1) // 2, )
        '''
        y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
        D_y = self.__pdist_euclidean(y_col_vec)
        d_y = self.__square_to_vec(D_y)
        y_contrasts = tf.cast(d_y == 0, tf.int32)
        return y_contrasts
