
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)

def variable_summaries(var, var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""  
  with tf.name_scope('summaries') and tf.device("/cpu:0"):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_' + var_name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev_'+var_name, stddev)
    tf.summary.scalar('max_'+var_name, tf.reduce_max(var))
    tf.summary.scalar('min_'+var_name, tf.reduce_min(var))
    tf.summary.histogram('histogram_'+var_name, var)
    
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  # if wd:
  #   weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  #   weight_decay.set_shape([])
  #   tf.add_to_collection('losses', weight_decay)
  return var

def conv_layer(inputs, kernel_size, stride, num_features,idx,dilation=1,padding="SAME",linear=False):

  
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.2, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))

    conv = tf.nn.convolution(inputs,weights,strides=[stride, stride],dilation_rate=[dilation,dilation],padding=padding)
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased, weights, biases
    #batchnorm should be before elu
    conv_rect_batch_norm = tf.contrib.layers.batch_norm(conv_biased,decay=0.999,epsilon=1e-5,scale=True,updates_collections=None)
    #conv_rect_batch_norm = conv_biased
    conv_rect = tf.nn.leaky_relu(conv_rect_batch_norm,name='{0}_conv'.format(idx), alpha=0.2)
    #conv_rect = tf.nn.relu(conv_rect_batch_norm)
    return conv_rect, weights, biases



def fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    inputs_processed = tf.nn.dropout(inputs_processed, keep_prob=0.5)
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.2, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(FLAGS.weight_init))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc'), weights, biases
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.leaky_relu(ip,name=str(idx)+'_fc', alpha=0.2), weights, biases




def deconv_layer(inputs, kernel_size,output_shape,idx, stride=2,padding="SAME",batch_norm=True,linear=False):
    with tf.variable_scope('{0}_deconv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]
      
      num_features =  output_shape[-1]
      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels],stddev=0.2, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))

      conv2d_transpose = tf.nn.conv2d_transpose(inputs, weights, output_shape, [1, stride, stride, 1], padding="SAME") + biases 
      if linear==True:
        return conv2d_transpose,weights,biases
      if batch_norm==True:
        conv_rect_batch_norm = tf.contrib.layers.batch_norm(conv2d_transpose,decay=0.999,epsilon=1e-5,scale=True,updates_collections=None)
        conv_rect_batch_norm = tf.nn.leaky_relu(conv_rect_batch_norm,alpha=0.2)
      return conv_rect_batch_norm,weights,biases

