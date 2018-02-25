
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2
import random

import layer_def as ld
import BasicConvLSTMCell
import glob
from scipy.misc import imread, imsave, imresize
import sys
sys.path.append(os.path.abspath('..'))
#print sys.path
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")


tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")


tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

FPS = 6
BATCH_SIZE = 10
IMAGE_SHAPE = 224
IMAGE_CHANNELS = 3
SEQ_LENGTH = 30
LEARNING_RATE = 0.002
LABEL_WEIGHT = 1
COLOR_FILTERS = 1
#mean_image = np.zeros((224,224,3))


def get_images(frames, mean_image, shape=IMAGE_SHAPE,use_mean_image):
  image_rows = shape
  image_columns = shape
  image_channels = IMAGE_CHANNELS

  images = [(imread(element)) for element in frames]
  if use_mean_image:
    images = np.asarray(images) - mean_image
  #print "mean image sum: " + str(np.sum(mean_image))

  images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,image_channels)
  print images.shape
  return images



def down_sampling_decoders(inputs,scope,hidden_1,hidden_2):
    #28 X 28 X 128


    #DECODER BLOCK 1
    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 128, dilation=1, "decode_1_1",padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")
    conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 256, dilation=1, "decode_1_2",padding="SAME")
    ld.variable_summaries(conv1_2, scope + "_conv1_2")
    #7 X 7 X 256

    # conv lstm cell 1
    shape = tf.shape(conv1_2)
    with tf.variable_scope("conv_lstm_1", initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 256)
      y_1, hidden_feature_map_1, matrix_1 = cell(conv1_2, hidden_1)
      ld.variable_summaries(y_1, scope + "_y_1")
      ld.variable_summaries(hidden_feature_map_1, scope + "_hidden_feature_map[1]")

    ld.variable_summaries(y_1, scope + "_y_1")

    y_1 += conv1_2
    ld.variable_summaries(y_1, scope + "_skip_y_1")



    #DECODER BLOCK 2
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(y_1, 3, 1, 256, dilation=2,"decode_2_1",padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")
    conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 512,  dilation=2,"decode_2_2",padding="SAME")
    ld.variable_summaries(conv2_2, scope + "_conv2_2")

    # conv lstm cell 2
    shape = tf.shape(conv2_2)
    with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 512)
      y_2, hidden_feature_map_2, matrix_2 = cell(conv2_2, hidden_2)
      ld.variable_summaries(y_2, scope + "_y_2")
      ld.variable_summaries(hidden_feature_map_2, scope + "_hidden_feature_map[2]")

    ld.variable_summaries(y_2, scope + "_y_2")
    #2 X 2 X 512

    y_2 += conv2_2
    ld.variable_summaries(y_2, scope + "_skip_y_2")

    shape = tf.shape(y_2)
    y_2 = tf.reshape(y_2,[shape[0],1,1,np.prod(shape[1:4])])
    # 1 X 1 X 2048
    
    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(y_2, 1, 1, 512,"decode_3_1",padding="SAME")
    # 1 X 1 X 512
    conv4_1, conv_w4_1, conv_b4_1 = ld.conv_layer(conv3_1, 1, 1, 1,"decode_4_1",padding="SAME")
    # 1 X 1 X 1

    output = tf.reshape(conv4_1,[-1])

    ld.variable_summaries(conv3_1, scope+ "_conv3_1")
    ld.variable_summaries(output, scope+ "_output")

    return output,hidden_feature_map_0, hidden_feature_map_1



def up_sampling_decoder(inputs,skip_connection_1,skip_connection_2,scope):
    #28 X 28 X 128 

    #DECODER BLOCK 1
    shape = tf.shape(inputs)
    output_shape = [shape[0],56,56,64]
    conv1_1, conv_w1_1, conv_b1_1 = ld.deconv_layer(inputs, 3, output_shape,"decode_1_1",stride=1,padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")
    #56 X 56 X 64
    
    conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 1, 64,dilation=1,"decode_1_2", padding="SAME")
    ld.variable_summaries(conv1_2, scope + "_conv1_2")
    #56 X 56 X 64

    conv1_2 += skip_connection_1
    ld.variable_summaries(conv1_2, scope + "_skip_conv1_2")


    #DECODER BLOCK 2
    shape = tf.shape(conv1_2)
    output_shape = [shape[0],112,112,32]
    conv2_1, conv_w2_1, conv_b2_1 = ld.deconv_layer(conv1_2, 3, output_shape,"decode_2_1",stride=1,padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")
    #112 X 112 X 32

    
    conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 1, 32,dilation=1,"decode_2_2", padding="SAME")
    ld.variable_summaries(conv2_2, scope + "_conv2_2")
    #112 X 112 X 32

    conv2_2 += skip_connection_2
    ld.variable_summaries(conv2_2, scope + "_skip_conv2_2")


    #DECODER BLOCK 3
    shape = tf.shape(conv2_2)
    output_shape = [shape[0],224,224,16]
    conv3_1, conv_w3_1, conv_b3_1 = ld.deconv_layer(conv2_2, 3, output_shape,"decode_3_1",stride=1,padding="SAME")
    ld.variable_summaries(conv3_1, scope + "_conv3_1")
    #224 X 224 X 16

    
    conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 1, COLOR_FILTERS,dilation=1,"decode_3_2", padding="SAME")
    ld.variable_summaries(conv3_2, scope + "_conv3_2")
    #224 X 224 X COLOR_FILTERS


    return conv3_2




def network(inputs, hidden_1,hidden_2,hidden_3, lstm=True):

    #ENCODER BLOCK 1
    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 1, 16, dilation=1, "encode_1_1",padding="SAME")
    ld.variable_summaries(conv1_1, "conv1_1")
    conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 32, dilation=1, "encode_1_2",padding="SAME")
    ld.variable_summaries(conv1_2, "conv1_2")

    # conv lstm cell 1
    shape = tf.shape(conv1_2)
    with tf.variable_scope('conv_lstm_1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 32)
      y_1, hidden_feature_map_1, matrix_1 = cell(conv1_2, hidden_1)
      ld.variable_summaries(y_1, "y_1")
      ld.variable_summaries(hidden_feature_map_1, "hidden_feature_map[1]")

    ld.variable_summaries(y_1, "y_1")
    #112 X 112 X 32


    lstm_1 = y_1
    y_1 += conv1_2
    ld.variable_summaries(y_1, "skip_y_1")


    #ENCODER BLOCK 2
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(y_1, 3, 1, 32, dilation=2,"encode_2_1",padding="SAME")
    ld.variable_summaries(conv2_1, "conv2_1")
    conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 64,  dilation=2,"encode_2_2",padding="SAME")
    ld.variable_summaries(conv2_2, "conv2_2")

    # conv lstm cell 2
    shape = tf.shape(conv2_2)
    with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
      y_2, hidden_feature_map_2, matrix_2 = cell(conv2_2, hidden_2)
      ld.variable_summaries(y_2, "y_2")
      ld.variable_summaries(hidden_feature_map_2, "hidden_feature_map[2]")

    ld.variable_summaries(y_2, "y_2")
    #56 X 56 X 64

    lstm_2 = y_2
    y_2 += conv2_2
    ld.variable_summaries(y_2, "skip_y_2")


    #ENCODER BLOCK 3
    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(y_2, 3, 2, 64, dilation=4,"encode_3_1",padding="SAME")
    ld.variable_summaries(conv3_1, "conv3_1")
    conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 2, 128,dilation=4,"encode_3_2", padding="SAME")
    ld.variable_summaries(conv3_2, "conv3_2")

    # conv lstm cell 3
    shape = tf.shape(conv3_2)
    with tf.variable_scope('conv_lstm_3', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 128)
      y_3, hidden_feature_map_3, matrix_3 = cell(conv3_2, hidden_3)
      ld.variable_summaries(y_3, "y_3")
      ld.variable_summaries(hidden_feature_map_3, "hidden_feature_map[3]")

    ld.variable_summaries(y_3, "y_3")
    #28 x 28 x 128

    lstm_3 = y_3
    y_3 += conv3_2
    ld.variable_summaries(y_3, "skip_y_3")

    '''
    with tf.variable_scope("frame_importance", initializer = tf.random_uniform_initializer(-.01, 0.1)):
      frame_importance_output = down_sampling_decoders(y_3,"frame_importance")

    with tf.variable_scope("segmentation", initializer = tf.random_uniform_initializer(-.01, 0.1)):
      segmentation_output = down_sampling_decoders(y_3,"segmentation")
    '''

    with tf.variable_scope("prednet", initializer = tf.random_uniform_initializer(-.01, 0.1)):
      prednet_output = up_sampling_decoder(y_3,y_2,y_1,"prednet")

      
    wts_list = [conv_w1_1, conv_b1_1, conv_w1_2, conv_b1_2, conv_w2_1, conv_b2_1,
    conv_w2_2, conv_b2_2, conv_w3_1, conv_b3_1, conv_w3_2, conv_b3_2, matrix_1, matrix_2, matrix_3]

    feature_map_list = [conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,y_1,y_2,y_3,lstm_1,lstm_2,lstm_3]


    return prednet_output,hidden_feature_map_0, hidden_feature_map_1, hidden_feature_map_2, hidden_feature_map_3,wts_list,feature_map_list



# make a template for reuse
network_template = tf.make_template('network', network)






def train():
  """Train ring_net for a number of steps."""


  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])
    labels = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])
    hidden_placeholder_1 = tf.placeholder(tf.float32,[BATCH_SIZE,112,112,32])
    hidden_placeholder_2 = tf.placeholder(tf.float32,[BATCH_SIZE,56,56,64])
    hidden_placeholder_3 = tf.placeholder(tf.float32,[BATCH_SIZE,28,28,128])
    
    x_dropout = x

    x_unwrap = []

    wts_list = []
    feature_map_list = []
    device_count = 0
    with tf.device("/gpu:" + str(device_count)):
        x_1, hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3,_,_ = network_template(x_dropout[:,0,:,:,:], 
          hidden_0 = hidden_placeholder_1,
          hidden_1 = hidden_placeholder_2,
          hidden_2 = hidden_placeholder_3)
        x_unwrap.append(x_1)

    gpu_devices = [i for i in range(0,8)]
    for i in xrange(1,SEQ_LENGTH):
        with tf.device("/gpu:" + str(device_count)):
          x_1, hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3, wts_list, feature_map_list = network_template(x_dropout[:,i,:,:,:], 
            hidden_0 = hidden_feature_map_1,
            hidden_1 = hidden_feature_map_2,
            hidden_2 = hidden_feature_map_3)
          x_unwrap.append(x_1)
        device_count+=1
        device_count%=1

    x_unwrap = tf.reshape(x_unwrap,[-1,SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])
    tf.summary.image('output_image',x_unwrap)
    x_unwrap = tf.reshape(x_unwrap,[-1])
    labels = tf.reshape(labels,[-1])

    print "LABELS SHAPE: " + str(labels.shape)
    print "X_UNWRAP SHAPE: " + str(x_unwrap.shape)

    x_unwrap /= 16
    labels   /= 16

    softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x_unwrap)
    
    print "LOSS SHAPE: "  + str(softmax_loss.shape)
    loss = tf.reduce_sum(softmax_loss)/BATCH_SIZE

    
    grad_list = tf.gradients(loss, wts_list)
    grad_feature_map_list = tf.gradients(loss, feature_map_list)

    wts_list_var_name = ["conv_w1_1","conv_b1_1", "conv_w1_2", "conv_b1_2", "conv_w2_1", "conv_b2_1",
  "conv_w2_2", "conv_b2_2", "conv_w3_1", "conv_b3_1", "conv_w3_2", "conv_b3_2", "matrix_1", "matrix_2", "matrix_3"]

    feature_map_list_name = ["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2",
    "y_1","lstm_1","y_2","lstm_2","y_3", "lstm_3"]

    for grad_num in range(0, len(grad_list_var_name)):
      ld.variable_summaries(grad_list[grad_num], "grad_" + wts_list_var_name[grad_num])
      ld.variable_summaries(wts_list[grad_num], wts_list_var_name[grad_num])

    for grad_num in range(0,len(feature_map_list_name)):
      ld.variable_summaries(grad_feature_map_list[grad_num], "grad_"+feature_map_list_name[grad_num])


    print "LOSS SHAPE: "  + str(loss.shape)
    
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, colocate_gradients_with_ops=True)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)


    #Loading the session
    print(" [*] Reading checkpoint...")
    checkpoint_dir = "./checkpoints/train_store_conv_lstm"
    model_name = "model.ckpt"

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print("ckpt: " + str(ckpt))
    if ckpt and ckpt.model_checkpoint_path:
        print("Trying to load the session")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print "ckpt_name: " + str(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("Session Loaded")
    else:
        print("No session to load")



    # Summary op
    graph_def = sess.graph
    print "graph_def: " + str(graph_def)
    tf_tensorboard = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./checkpoints/train_store_conv_lstm",graph_def)
    summary_writer_it = 0


    MAX_EPOCHS = 10000000
    epoch = 0
    videos = glob.glob("../dataset/video_frames/Webscope_I4_6/[!.]*")
    videos = videos[0:1]

    saver_step = 0

    mean_image = np.load("mean_image.npy")

    while(epoch<MAX_EPOCHS):
      print("Epoch: " + str(epoch))
      random.shuffle(videos)
      print "Len of videos glob: " + str(len(videos))
      start_count = 0

      #loading the batches
      while start_count+BATCH_SIZE<len(videos):
        print "Loading the batches"
        videos_in_iteration = videos[start_count: start_count+BATCH_SIZE]
        start_count += BATCH_SIZE
        video_index = 0
        frame = dict()
        mini_len = 1000000
        
        #Load glob of each video in mini batch
        for video in videos_in_iteration:
          print "Load glob of each video in mini batch"
          frame[video_index] = glob.glob(video+"/*.jpg")
          
          mini_len = min(mini_len, len(frame[video_index]))
          video_index += 1

        #Operating on glob of each individual video
        frame_start_count = 0

        hidden_input_1 = np.zeros((BATCH_SIZE,112,112,32),dtype=np.float32)
        hidden_input_2 = np.zeros((BATCH_SIZE,56,56,64),dtype=np.float32)
        hidden_input_3 = np.zeros((BATCH_SIZE,28,28,128),dtype=np.float32)

        

        while frame_start_count+SEQ_LENGTH+1<mini_len:
          selected_batch = []
          for video_index_num in range(0, video_index):
            selected_frames = frame[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH]
            output_selected_frames = frame[video_index_num][frame_start_count+1: frame_start_count+SEQ_LENGTH+1]
            selected_batch.append(get_images(selected_frames,mean_image,1))
            output_selected_batch.append(get_images(output_selected_frames,mean_image,0))
            

          frame_start_count += SEQ_LENGTH
          input_data = np.asarray(selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
          prednet_output_data = np.asarray(output_selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
          print("input_data shape: " + str(input_data.shape))
          print("prednet_output_data shape: " + str(prednet_output_data.shape))
          



          t = time.time()
          #frame_importance_numpy = np.asarray(frame_importance[start_count:start_count+10]).reshape(-1)
          _, loss_r, summary, model_output, hidden_last_batch_1, hidden_last_batch_2, hidden_last_batch_3 = sess.run([train_op, loss, tf_tensorboard, output,hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3],
            feed_dict={x:input_data, labels:prednet_output_data,
            hidden_placeholder_1:hidden_input_1,
            hidden_placeholder_2:hidden_input_2,
            hidden_placeholder_3:hidden_input_3})

          elapsed = time.time() - t

          hidden_input_1 = hidden_last_batch_1
          hidden_input_2 = hidden_last_batch_2
          hidden_input_3 = hidden_last_batch_3


          # Write data to tensorboard
          summary_writer.add_summary(summary, summary_writer_it)
          summary_writer_it += 1

          print "LOSS: " + str(loss_r)
          
          print("Iteration: " + str(saver_step))
          saver_step+=1

                  
          if saver_step%50 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=saver_step)  
            print("saved to " + FLAGS.train_dir)
            


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()










        







