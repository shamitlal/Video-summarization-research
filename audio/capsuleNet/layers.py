import tensorflow as tf

def variable_summaries(var, var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_' + var_name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev_'+var_name, stddev)
    tf.summary.scalar('max_'+var_name, tf.reduce_max(var))
    tf.summary.scalar('min_'+var_name, tf.reduce_min(var))
    tf.summary.histogram('histogram_'+var_name, var)

def create_conv(layer_num, prev, shape, padding='VALID', strides=[1, 1, 1, 1], relu=False,
                 max_pooling=False, mp_ksize=[1, 2, 2, 1], mp_strides=[1, 2, 2, 1]):
    """
        Create a convolutional layer with relu and/mor max pooling(Optional)
    """
    conv_w = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1))
    variable_summaries(conv_w,"conv_w_"+layer_num)
    conv_b = tf.Variable(tf.ones(shape[-1])*0.01)
    variable_summaries(conv_b,"conv_b_"+layer_num)
    conv   = tf.nn.conv2d(prev, conv_w, strides=strides, padding=padding) + conv_b

    if relu:
        conv = tf.nn.relu(conv)

    if max_pooling:
        conv = tf.nn.max_pool(conv, ksize=mp_ksize, strides=mp_strides, padding='VALID')

    return conv, conv_w, conv_b

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