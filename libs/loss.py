import tensorflow as tf
import math
import numpy as np
from tensorflow.python.keras import layers
import tensorflow.keras.backend as K
W_INIT = tf.contrib.layers.xavier_initializer(uniform=False)
regularizer = tf.contrib.layers.l2_regularizer
from tensorflow.python.keras import regularizers
def get_call_func(y_true, y_pred, config):
    val = np.random.laplace(size=[config["embed_num"], config['class_num']])
    weights = K.variable(value=val, name='classify_weight', dtype=tf.float32)
    if config['loss_type'] == 'arcface':
        y_pred = K.l2_normalize(y_pred)
        cons_t = layers.Dense(config['class_num'], use_bias=False, kernel_regularizer=regularizers.l2(0.01))(y_pred)
    elif config['loss_type'] == 'softmax':
        logits = layers.Dense(config['class_num'], kernel_initializer=W_INIT,
                              kernel_regularizer=regularizer(config['weight_decay']))(y_pred)
    else:
        raise ValueError('Invalid loss type.')
    if config['loss_type'] == 'arcface':
        logits = arcface_logits(y_pred, weights, y_true, config['class_num'], config['logits_scale'],
                                config['logits_margin'])

    def loss_func(_, __):
        inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
        train_loss = inference_loss + tf.losses.get_regularization_loss()
        return train_loss
    return loss_func, logits


def arcface_logits(cos_t, labels, class_num, s, m):

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val) 
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
    return output
