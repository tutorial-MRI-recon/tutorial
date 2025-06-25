import tensorflow as tf
import numpy as np


def conv_layer(input_data, conv_filter, is_relu=False, is_scaling=False):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    # W = tf.get_variable('W', shape=conv_filter, initializer=tf.random_normal_initializer(0, 0.05))
    W = tf.compat.v1.get_variable('W', shape=conv_filter, initializer=tf.compat.v1.keras.initializers.he_normal())

    x = tf.nn.conv2d(input=input_data, filters=W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.keras.layers.BatchNormalization(name='BN')(x)
    # x = tf.keras.layers.Dropout(rate=0.01)(x)

    if (is_relu):
        # x = tf.nn.relu(x)
        x = tf.nn.leaky_relu(x, alpha=0.05)

    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)

    return x


def ResNet(input_data, nb_res_blocks, nb_echo):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """
    num_chan = 128 # 128

    conv_filters = dict([('w1', (3, 3, nb_echo * 2, num_chan)), ('w2', (3, 3, num_chan, num_chan)), ('w3', (3, 3, num_chan, nb_echo * 2))])
    intermediate_outputs = {}

    with tf.compat.v1.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer(input_data, conv_filters['w1'], is_relu=True, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.compat.v1.variable_scope('ResBlock' + str(i)):
            intermediate_outputs['layer' + str(i)] = conv_layer(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2'], is_relu=True, is_scaling=False)

    with tf.compat.v1.variable_scope('LastLayer'):
        rb_output = conv_layer(intermediate_outputs['layer' + str(i)], conv_filters['w3'], is_relu=False, is_scaling=False)

    with tf.compat.v1.variable_scope('Residual'):
        nw_output = rb_output + input_data # org
        # nw_output = rb_output # mod

    return nw_output


def mu_param():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        mu = tf.compat.v1.get_variable(name='mu', dtype=tf.float32, initializer=0.05, trainable=True) # default: .05

    return mu
