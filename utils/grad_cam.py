# Grad-CAM for tf-keras

import cv2
import numpy as np
import tensorflow as tf

tfk = tf.keras

# def get_grad_cam(model, convLayerIdx, x, width, height):
#     # rescale
#     rescaledX = x
#     pred = model.predict(rescaledX)
#     classIdx = np.argmax(pred, axis=1)
    
#     relu = np.vectorize(lambda x: max(0., x))

#     jetCmapList = []
#     for idx, (iX, ci) in enumerate(zip(rescaledX, classIdx)):
#         classOutput = model.layers[-1].output[:, ci]
#         convOutput = model.layers[convLayerIdx].output

#         grads = K.gradients(classOutput, convOutput)[0]
#         getGrad = K.function([model.layers[0].input, K.learning_phase()], [convOutput, grads])

#         iX = iX[np.newaxis, :, :, :]

#         # learning_phase    0: test, 1: train
#         output, gradients = getGrad([iX, 0])
#         output, gradients = output[0], gradients[0]

#         alpha = np.mean(gradients, axis=(0, 1))
#         gcmap = np.dot(output, alpha)  # output, alphaの順ならOK
#         gcmap = relu(gcmap)
#         gcmap /= np.max(gcmap)

#         resizedCmap = cv2.resize(gcmap, (width, height), cv2.INTER_LINEAR)
#         jetCmap = cv2.applyColorMap(np.uint8(255 * resizedCmap), cv2.COLORMAP_JET)
#         jetCmap = cv2.cvtColor(jetCmap, cv2.COLOR_BGR2RGB)

#         jetCmapList.append(jetCmap)

#     return jetCmapList, classIdx



@tf.function
def _f(model, inputs, labels, loss_fn, final_conv_idx)
    intermediate = model.get_layer(index=final_conv_idx)
    tmp_model = tfk.Model(model.input, outputs=[model.output, intermediate])
    with tf.GradientTape() as tape:
        # output = model.layers[-1].output
        pred, conv_out = tmp_model(inputs)
        # conv_output = model.get_layer(index=final_conv_idx)(x)
        loss_val = loss_fn(labels, pred)

        grads = tape.gradient(loss_val, tmp_model.trainable_variables)
        final_conv_grad = grads[final_conv_idx]
    
    return pred, final_conv_grad
    

def get_grad_cam(model, inputs, labels, loss_fn, final_conv_idx):
    pred, final_conv_grad = _f(model, inputs, labels, loss_fn, final_conv_grad)
    pred, final_conv_grad = pred.numpy(), final_conv_grad.numpy()

    cam_list = []
    relu = np.vectorize(lambda x: max(0, x))
    width, height = inputs.shape[:2]
    for out, grad in zip(pred, final_conv_grad):
        alpha = np.mean(grad, axis=(0, 1))
        gcam = np.dot(out, alpha)  # output, alphaの順ならOK
        gcam = relu(gcam)
        if np.min(gcam) == 0 and np.max(gcam) == 0:
            gcam[...] = 0
        else:
            gcam = (gcam - np.min(gcam)) / (np.max(gcam) - np.min(gcam))

        # opencvではshapeが(height, width, ch)として扱われる
        # cv2.resize(img, (width, height), fileter)
        resized_gcam = cv2.resize(gcam, (width, height), cv2.INTER_LINEAR)
        cam_list.append(resized_gcam)

    return cam_list

