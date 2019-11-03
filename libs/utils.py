import tensorflow as tf
from tensorflow.python.keras import layers
import libs.models.resnet_v2 as resnet

import os
import cv2
import io

import math
import numpy as np

from sklearn.model_selection import KFold
from scipy import interpolate

def get_model_by_config(input_shape, y_true, y_pred, config):
    import tensorflow.keras.backend as K
    from libs.loss import arcface_logits

    if config['out_type'] == 'E':
        def layers_fn(x):
            bn_axis = -1
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                          name='conv3_bn')(x)
            x = layers.Dropout(rate=config['keep_prob'])(x)
            x = layers.Flatten()(x)
            x = layers.Dense(config['embd_size'])(x)
            x = layers.BatchNormalization(scale=False, axis=bn_axis, epsilon=1.001e-5,
                                          name='conv2_bn')(x)
            from tensorflow.python.keras import regularizers
            if config['loss_type'] == 'arcface':
                x = K.l2_normalize(x)
                x = layers.Dense(config['class_num'], use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="arcface")(x)
            elif config['loss_type'] == 'softmax':
                x = layers.Dense(config['class_num'], use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="softmax")(x)
            else:
                raise ValueError('Invalid loss type.')
            return x
        call_fn = layers_fn
    else:
        raise ValueError('Invalid out type.')

    model = resnet.ResNet50V2(config, input_shape=input_shape, weights=None, layers_fn=call_fn)
    def loss_func(_, __):
        logits = model.output
        if config['loss_type'] == 'arcface':
            logits = arcface_logits(logits, y_true, config['class_num'], config['logits_scale'],
                               config['logits_margin'])
        inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
        train_loss = inference_loss + tf.losses.get_regularization_loss()
        return train_loss


    return model, loss_func

def check_folders(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tensor_description(var):
    """Returns a compact and informative string about a tensor.
    Args:
        var: A tensor variable.
    Returns:
        a string with type and size, e.g.: (float32 1x8x8x1024).
    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description

def get_port(num=1):
    def tryPort(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = False
        try:
            sock.bind(("0.0.0.0", port))
            result = True
        except:
            pass
        sock.close()
        return result

    port = []
    for i in range(1024, 65535):
        if tryPort(i):
            port.append(i)
            if len(port) == num:
                return port
