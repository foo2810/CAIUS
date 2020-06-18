# Grad-CAM for tf-keras

import cv2
import numpy as np
import tensorflow as tf

tfk = tf.keras

@tf.function
def _f(model, inputs, label, loss_fn, final_conv_idx):
    with tf.GradientTape() as tape:
        # pred = model(inputs)
        pred, conv_out = model(inputs)
        # loss_val = loss_fn(labels, pred)
        # grads = tape.gradient(loss_val, conv_out)
        grads = tape.gradient(pred[:, label], conv_out)
        final_conv_grad = grads[final_conv_idx]
    del tape
    
    # conv_out = model.layers[final_conv_idx](inputs)
    
    return pred, conv_out, final_conv_grad
    

def get_grad_cam(model, inputs, label, loss_fn, final_conv_idx):
    # if final_conv_idx < 0:
    #     final_conv_idx += len(model.layers)
    h = model.layers[final_conv_idx].output

    tmp_model = tfk.Model(model.inputs, [model.output, h])
    pred, conv_out, final_conv_grad = _f(tmp_model, inputs, label, loss_fn, final_conv_idx)
    pred, conv_out, final_conv_grad = tf.cast(pred, tf.float32), tf.cast(conv_out, tf.float32), tf.cast(final_conv_grad, tf.float32)
    pred, conv_out, final_conv_grad = pred.numpy(), conv_out.numpy(), final_conv_grad.numpy()

    cam_list = []
    relu = np.vectorize(lambda x: max(0, x))
    width, height = inputs.shape[1:3]
    for out, c_out in zip(pred, conv_out):
        # alpha = tf.reduce_mean(final_conv_grad, axis=(0, 1))
        # gcam = tf.tensordot(c_out, alpha, axes=[2, 0])
        # gcam = tfk.activations.relu(gcam)
        # if tf.reduce_min(gcam) == tf.zeros([]) and tf.reduce_max(gcam) == tf.zeros([]):
        #     gcam *= tf.constant(0, dtype=tf.float32)
        # else:
        #     gcam = (gcam - tf.reduce_min(gcam)) / (tf.reduce_max(gcam) - tf.reduce_min(gcam))
        # gcam = tf.transpose(gcam, perm=[1, 0])

        alpha = np.mean(final_conv_grad, axis=(0, 1))
        gcam = np.dot(c_out, alpha)  # output, alphaの順ならOK
        gcam += 0.5
        gcam = relu(gcam)
        if np.min(gcam) == 0 and np.max(gcam) == 0:
            gcam[...] = 0
        else:
            gcam = (gcam - np.min(gcam)) / (np.max(gcam) - np.min(gcam))

        # opencvではshapeが(height, width, ch)として扱われる
        # cv2.resize(img, (width, height), fileter)
 
        # gcam = np.transpose(gcam, (1, 0, 2))[..., 0]
        # resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)

        resized_gcam = gcam
        cam_list.append(resized_gcam)

    return cam_list

