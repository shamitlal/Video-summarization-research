import tensorflow as tf

def create_conv(prev, shape, padding='VALID', strides=[1, 1, 1, 1], relu=False,
                 max_pooling=False, mp_ksize=[1, 2, 2, 1], mp_strides=[1, 2, 2, 1]):
    """
        Create a convolutional layer with relu and/mor max pooling(Optional)
    """
    conv_w = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1,  seed=0))
    conv_b = tf.Variable(tf.zeros(shape[-1]))
    conv   = tf.nn.conv2d(prev, conv_w, strides=strides, padding=padding) + conv_b

    if relu:
        conv = tf.nn.relu(conv)

    if max_pooling:
        conv = tf.nn.max_pool(conv, ksize=mp_ksize, strides=mp_strides, padding='VALID')

    return conv

def fc(prev, input_size, output_size, relu=False, sigmoid=False, no_bias=False,
        softmax=False):
    """
        Create fully connecter layer with relu(Optional)
    """
    fc_w = tf.Variable(
        tf.truncated_normal(shape=(input_size, output_size), mean = 0., stddev = 0.1))
    fc_b = tf.Variable(tf.zeros(output_size))
    pre_activation = tf.matmul(prev, fc_w)
    activation = None

    if not no_bias:
        pre_activation = pre_activation + fc_b
    if relu:
        activation = tf.nn.relu(pre_activation)
    if sigmoid:
        activation = tf.nn.sigmoid(pre_activation)
    if softmax:
        activation = tf.nn.softmax(pre_activation)

    if activation is None:
        activation = pre_activation

    return activation, pre_activation