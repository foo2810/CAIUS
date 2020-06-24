# Grad-CAM for tf-keras

import cv2
import numpy as np
import tensorflow as tf

tfk = tf.keras

@tf.function
def _f(model, inputs, label, loss_fn, final_conv_idx):
    with tf.GradientTape() as tape:
        # pred = model(inputs)
        pred, conv_out = model(inputs, training=False)
        # loss_val = loss_fn(labels, pred)
        # grads = tape.gradient(loss_val, conv_out)
        grads = tape.gradient(pred[:, label], conv_out)
    del tape
    
    # conv_out = model.layers[final_conv_idx](inputs)
    
    return pred, conv_out, grads
    

def get_grad_cam(model, inputs, label, loss_fn, final_conv_idx=None, final_conv_name=None, conv_layer=None):
    if conv_layer is not None:
        h = conv_layer
    else:
        if final_conv_idx is None and final_conv_name is None:
            raise ValueError
        elif final_conv_idx is not None and final_conv_name is None:
            h = model.layers[final_conv_idx].output
        elif final_conv_idx is None and final_conv_name is not None:
            h = model.get_layer(name=final_conv_name).output
        else:
            # 名前指定を優先
            h = model.get_layer(name=final_conv_name).output

    tmp_model = tfk.Model(model.inputs, [model.output, h])
    pred, conv_out, final_conv_grad = _f(tmp_model, inputs, label, loss_fn, final_conv_idx)
    pred, conv_out, final_conv_grad = tf.cast(pred, tf.float32), tf.cast(conv_out, tf.float32), tf.cast(final_conv_grad, tf.float32)
    pred, conv_out, final_conv_grad = pred.numpy(), conv_out.numpy(), final_conv_grad.numpy()

    cam_list = []
    width, height = inputs.shape[1:3]
    for out, c_out, c_grad in zip(pred, conv_out, final_conv_grad):
        # alpha = tf.reduce_mean(final_conv_grad, axis=(0, 1))
        # gcam = tf.tensordot(c_out, alpha, axes=[2, 0])
        # gcam = tfk.activations.relu(gcam)
        # if tf.reduce_min(gcam) == tf.zeros([]) and tf.reduce_max(gcam) == tf.zeros([]):
        #     gcam *= tf.constant(0, dtype=tf.float32)
        # else:
        #     gcam = (gcam - tf.reduce_min(gcam)) / (tf.reduce_max(gcam) - tf.reduce_min(gcam))
        # gcam = tf.transpose(gcam, perm=[1, 0])

        alpha = np.mean(c_grad, axis=(0, 1))
        gcam = np.dot(c_out, alpha)  # output, alphaの順ならOK
        gcam += 0.5
        gcam = np.maximum(gcam, 0)
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

    return cam_list, pred

def get_grad_cam_plusplus(model, inputs, label, loss_fn, conv_idx=None, conv_name=None, conv_layer=None):
    if conv_layer is not None:
        h = conv_layer
    else:
        if conv_idx is None and conv_name is None:
            raise ValueError
        elif conv_idx is not None and conv_name is None:
            h = model.layers[conv_idx].output
        elif conv_idx is None and conv_name is not None:
            h = model.get_layer(name=conv_name).output
        else:
            # 名前指定を優先
            h = model.get_layer(name=conv_name).output

    tmp_model = tfk.Model(model.inputs, [model.output, h])
    pred, conv_out, conv_grad = _f(tmp_model, inputs, label, loss_fn, conv_idx)
    pred, conv_out, conv_grad = tf.cast(pred, tf.float32), tf.cast(conv_out, tf.float32), tf.cast(conv_grad, tf.float32)
    pred, conv_out, conv_grad = pred.numpy(), conv_out.numpy(), conv_grad.numpy()

    cam_list = []
    width, height = inputs.shape[1:3]
    for out, c_out, c_grad in zip(pred, conv_out, conv_grad):
        y_c = out[label]

        conv_grad_first = np.exp(y_c) * c_grad
        conv_grad_second = np.exp(y_c) * c_grad * c_grad
        conv_grad_third = np.exp(y_c) * c_grad * c_grad * c_grad

        global_sum = np.sum(c_out.reshape((-1, c_out.shape[2])), axis=0)
        alpha_denom = conv_grad_second * 2.0 + conv_grad_third * global_sum.reshape((1, 1, c_out.shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1)#np.ones(alpha_denom.shape))

        alpha_num = conv_grad_second

        alphas = alpha_num / alpha_denom

        #alphas_norm_const = np.sum(alphas, axis=(0, 1))
        alphas_norm_const = np.sum(np.sum(alphas, axis=0), axis=0)
        alphas_norm_const = np.where(alphas_norm_const != 0.0, alphas_norm_const, 1)#np.ones(alphas_norm_const.shape))
        alphas /= alphas_norm_const

        weights = np.maximum(conv_grad_first, 0.0)
        dlw = np.sum((weights * alphas).reshape((-1, c_out.shape[2])), axis=0)    # deep linearization weights
        dlw = dlw.reshape((1, 1, -1))

        gcam = np.sum(dlw * c_out, axis=2)
        # gcam += 0.5
        gcam = np.maximum(gcam, 0.0)
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

    return cam_list, pred

