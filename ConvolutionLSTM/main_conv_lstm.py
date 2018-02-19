
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

FPS = 3
BATCH_SIZE = 10
IMAGE_SHAPE = 224
IMAGE_CHANNELS = 3
SEQ_LENGTH = 26
LEARNING_RATE = 0.002
LABEL_WEIGHT = 1
#mean_image = np.zeros((224,224,3))
def get_frame_importance(file_dir):
  f = open(file_dir,"r")
  video_to_frame_importance = dict()
  for video_imp in f:
    tab_separated_values = video_imp.split('\t')
    scores = tab_separated_values[2].split(',')
    i=0
    final_scores = [(float(score)-1)/2.0 for score in scores[::10]]
    '''
    gaussian_scores = np.random.normal(final_scores,0.3)
    final_scores_gaussian = []
    for score,score_gaussian in zip(final_scores,gaussian_scores):
      if score_gaussian < score:
        score_gaussian = max(score_gaussian,score-0.49)
      elif score_gaussian>score:
        score_gaussian = min(score_gaussian,score+0.49)
      final_scores_gaussian.append(score_gaussian)
    '''
    final_scores_gaussian = np.asarray(final_scores)
    video_to_frame_importance[tab_separated_values[0]]=final_scores_gaussian
    
  return video_to_frame_importance

def get_images(frames, mean_image, shape=IMAGE_SHAPE):
  image_rows = shape
  image_columns = shape
  image_channels = IMAGE_CHANNELS

  images = [(imread(element)) for element in frames]
  images = np.asarray(images) - mean_image
  #print "mean image sum: " + str(np.sum(mean_image))

  images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,image_channels)
  print images.shape
  return images

#label smoothening. Check if smoothening less important frames affect adversialy or not.
def convert_importance_to_vector(labels):
  final_importance_vector = []
  for element in labels:
    if element==0:
      element_array = np.asarray([0.6,0.25,0.15,0.0,0.0])
    elif element==1:
      element_array = np.asarray([0.20,0.6,0.20,0.0,0.0])
    elif element==2:
      element_array = np.asarray([0.0,0.30,0.60,0.10,0.0])
    elif element==3:
      element_array = np.asarray([0.0,0.0,0.10,0.60,0.30])
    elif element==4:
      element_array = np.asarray([0.0,0.0,0.0,0.35,0.65])
    final_importance_vector.append(element_array) 
  final_importance_vector = np.asarray(final_importance_vector).reshape((-1,5))
  return final_importance_vector

def compute_weights_array(labels):
  final_weight_vector = []
  for element in labels:
    if element==0:
      element_weight = 1
    elif element==1:
      element_weight = 1
    elif element==2:
      element_weight = 1
    elif element==3:
      element_weight = 4
    elif element==4:
      element_weight = 4
    else:
      element_weight = 1
    final_weight_vector.append(element_weight) 
  final_weight_vector = np.asarray(final_weight_vector).reshape((-1))
  return final_weight_vector



def network(inputs, hidden_0,hidden_1,hidden_2,hidden_3, lstm=True):


  conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 1, 16, "encode_1_1",padding="SAME")
  ld.variable_summaries(conv1_1, "conv1_1")
  conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 16, "encode_1_2",padding="SAME")
  ld.variable_summaries(conv1_2, "conv1_2")

  # conv2
  conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(conv1_2, 3, 1, 32, "encode_2_1",padding="SAME")
  ld.variable_summaries(conv2_1, "conv2_1")
  conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 32, "encode_2_2",padding="SAME")
  ld.variable_summaries(conv2_2, "conv2_2")

  # conv3
  conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(conv2_2, 3, 1, 64, "encode_3_1",padding="SAME")
  ld.variable_summaries(conv3_1, "conv3_1")
  conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 2, 64, "encode_3_2", padding="SAME")
  ld.variable_summaries(conv3_2, "conv3_2")
  # 28 x 28 x 64


  # conv lstm cell 1
  shape = tf.shape(conv3_2)
  with tf.variable_scope('conv_lstm_1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
    y_1, hidden_feature_map_0, matrix_1 = cell(conv3_2, hidden_0)
    ld.variable_summaries(y_1, "y_1")
    ld.variable_summaries(hidden_feature_map_0, "hidden_feature_map[0]")

  y_1_pool = tf.nn.max_pool(y_1, [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  ld.variable_summaries(y_1_pool, "y_1_pool")
  # 14 X 14 X 64


  # conv lstm cell 2
  shape = tf.shape(y_1_pool)
  with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 128)
    y_2, hidden_feature_map_1, matrix_2 = cell(y_1_pool, hidden_1)
    ld.variable_summaries(y_2, "y_2")
    ld.variable_summaries(hidden_feature_map_1, "hidden_feature_map[1]")

  y_2_pool = tf.nn.max_pool(y_2, [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  ld.variable_summaries(y_2_pool, "y_2_pool")
  #7 x 7 x 128


  # conv lstm cell 3
  shape = tf.shape(y_2_pool)
  with tf.variable_scope('conv_lstm_3', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 128)
    y_3, hidden_feature_map_2, matrix_3 = cell(y_2_pool, hidden_2)
    ld.variable_summaries(y_3, "y_3")
    ld.variable_summaries(hidden_feature_map_2, "hidden_feature_map[2]")

  y_3_pool = tf.nn.max_pool(y_3, [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  ld.variable_summaries(y_3_pool, "y_3_pool")
  #4 x 4 x 128


  # conv lstm cell 4
  shape = tf.shape(y_3_pool)
  with tf.variable_scope('conv_lstm_4', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 256)
    y_4, hidden_feature_map_3, matrix_4 = cell(y_3_pool, hidden_3)
    ld.variable_summaries(y_4, "y_4")
    ld.variable_summaries(hidden_feature_map_3, "hidden_feature_map[3]")

  y_4_pool = tf.nn.max_pool(y_4, [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  ld.variable_summaries(y_4_pool, "y_4_pool")
  #2 x 2 x 256
  
  fn1, fn1_w, fn1_b = ld.fc_layer(y_4_pool, 512, flat=True,idx="fc_1")
  fn2, fn2_w, fn2_b = ld.fc_layer(fn1, 128,idx="fc_2")
  output, output_w, output_b = (ld.fc_layer(fn2, 1, linear = True,idx="fc_3"))
  ld.variable_summaries(fn1, "fn1")
  ld.variable_summaries(fn2, "fn2")
  ld.variable_summaries(output, "output")
  wts_list = [conv_w1_1, conv_b1_1, conv_w1_2, conv_b1_2, conv_w2_1, conv_b2_1,
  conv_w2_2, conv_b2_2, conv_w3_1, conv_b3_1, conv_w3_2, conv_b3_2, matrix_1, matrix_2, matrix_3,
  matrix_4, fn1_w, fn1_b, fn2_w, fn2_b, output_w, output_b]

  feature_map_list = [conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,y_1,y_1_pool,y_2,y_2_pool,
  y_3, y_3_pool,y_4,y_4_pool,fn1,fn2,output]

  return output, hidden_feature_map_0, hidden_feature_map_1, hidden_feature_map_2, hidden_feature_map_3, wts_list, feature_map_list




# make a template for reuse
network_template = tf.make_template('network', network)




def train():
  """Train ring_net for a number of steps."""


  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE*SEQ_LENGTH])
    hidden_placeholder_1 = tf.placeholder(tf.float32,[BATCH_SIZE,28,28,128])
    hidden_placeholder_2 = tf.placeholder(tf.float32,[BATCH_SIZE,14,14,256])
    hidden_placeholder_3 = tf.placeholder(tf.float32,[BATCH_SIZE,7,7,256])
    hidden_placeholder_4 = tf.placeholder(tf.float32,[BATCH_SIZE,4,4,512])
    label_weights = tf.placeholder(tf.float32,[BATCH_SIZE*SEQ_LENGTH])
    # possible dropout inside
    x_dropout = x

    # create network
    x_unwrap = []

    # conv network
    wts_list = []
    feature_map_list = []
    device_count = 0
    with tf.device("/gpu:" + str(device_count)):
        x_1, hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3,hidden_feature_map_4,_,_ = network_template(x_dropout[:,0,:,:,:], 
          hidden_0 = hidden_placeholder_1,
          hidden_1 = hidden_placeholder_2,
          hidden_2 = hidden_placeholder_3,
          hidden_3 = hidden_placeholder_4)
        x_unwrap.append(x_1)

    gpu_devices = [i for i in range(0,8)]
    for i in xrange(1,SEQ_LENGTH):
        with tf.device("/gpu:" + str(device_count)):
          x_1, hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3,hidden_feature_map_4, wts_list, feature_map_list = network_template(x_dropout[:,i,:,:,:], 
            hidden_0 = hidden_feature_map_1,
            hidden_1 = hidden_feature_map_2,
            hidden_2 = hidden_feature_map_3,
            hidden_3 = hidden_feature_map_4)
          x_unwrap.append(x_1)
        device_count+=1
        device_count%=1


    '''
    #SHAPE OF X_WRAP : BATCH_SIZE * SEQ_LENGTH
    x_unwrap = tf.reshape(x_unwrap,[-1])

    print "LABELS SHAPE: " + str(labels)
    output_sigmoid = tf.nn.sigmoid(x_unwrap)
    output_sigmoid = tf.round(output_sigmoid)
    output_sigmoid_int = tf.cast(output_sigmoid,tf.int32)
    print "OUTPUT_SIGMOID SHAPE: " + str(output_sigmoid)
    correct_prediction = tf.equal(output_sigmoid, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    


    sigmoid_loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=x_unwrap,pos_weight=LABEL_WEIGHT)
    print "LOSS SHAPE: "  + str(sigmoid_loss.shape)
    loss = tf.reduce_sum(sigmoid_loss)/BATCH_SIZE
    '''


    x_unwrap = tf.reshape(x_unwrap,[-1])

    print "LABELS SHAPE: " + str(labels)
    print "X_UNWRAP SHAPE: " + str(x_unwrap)
    output = tf.round(x_unwrap)
    output_integer = tf.cast(output,tf.int64)
    correct_prediction = tf.equal(output_integer, tf.cast(labels,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #sigmoid_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=x_unwrap)
    #mse_loss = tf.squared_difference(labels,x_unwrap)
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=x_unwrap)
    #weighted_sigmoid_loss = sigmoid_loss * label_weights
    print "LOSS SHAPE: "  + str(sigmoid_loss.shape)
    loss = tf.reduce_sum(sigmoid_loss)/BATCH_SIZE

    grad_list = tf.gradients(loss, wts_list)
    grad_feature_map_list = tf.gradients(loss, feature_map_list)

    grad_list_var_name = ["conv_w1_1","conv_b1_1", "conv_w1_2", "conv_b1_2", "conv_w2_1", "conv_b2_1",
  "conv_w2_2", "conv_b2_2", "conv_w3_1", "conv_b3_1", "conv_w3_2", "conv_b3_2", "matrix_1", "matrix_2", "matrix_3",
  "matrix_4", "fn1_w", "fn1_b", "fn2_w", "fn2_b", "output_w", "output_b"]

    feature_map_list_name = ["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2",
    "y_1","y_1_pool","y_2","y_2_pool","y_3", "y_3_pool","y_4","y_4_pool","fn1","fn2","output"]

    for grad_num in range(0, len(grad_list_var_name)):
      ld.variable_summaries(grad_list[grad_num], "grad_" + grad_list_var_name[grad_num])
      ld.variable_summaries(wts_list[grad_num], grad_list_var_name[grad_num])

    for grad_num in range(0,len(feature_map_list_name)):
      ld.variable_summaries(grad_feature_map_list[grad_num], "grad_"+feature_map_list_name[grad_num])


    print "LOSS SHAPE: "  + str(loss.shape)
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

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


    # checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    # saver.save(sess, checkpoint_path, global_step=1)  
    # print("saved to " + FLAGS.train_dir)


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
    videos = glob.glob("../dataset/video_frames/Webscope_I4/[!.]*")

    labels_map = get_frame_importance("../dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv")
    #loading the epochs
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
        this_frame_importance = dict()
        #Load glob of each video in mini batch
        for video in videos_in_iteration:
          print "Load glob of each video in mini batch"
          frame[video_index] = glob.glob(video+"/*.jpg")
          #print "frame[video_index]=" + str(frame[video_index])
          this_frame_importance[video_index] = labels_map[video.split('/')[-1]]
          mini_len = min(mini_len, len(frame[video_index]))
          video_index += 1

        print "this_frame_importance shape: " + str(len(this_frame_importance.keys()))
        #Operating on glob of each individual video
        frame_start_count = 0

        hidden_input_1 = np.zeros((BATCH_SIZE,28,28,128),dtype=np.float32)
        hidden_input_2 = np.zeros((BATCH_SIZE,14,14,256),dtype=np.float32)
        hidden_input_3 = np.zeros((BATCH_SIZE,7,7,256),dtype=np.float32)
        hidden_input_4 = np.zeros((BATCH_SIZE,4,4,512),dtype=np.float32)

        

        while frame_start_count+SEQ_LENGTH<mini_len:
          selected_batch = []
          selected_importance_labels = []
          for video_index_num in range(0, video_index):
            selected_frames = frame[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH]
            selected_batch.append(get_images(selected_frames,mean_image))
            selected_importance_labels.append(this_frame_importance[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH])


          frame_start_count += SEQ_LENGTH
          dat = np.asarray(selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
          dat_label = np.asarray(selected_importance_labels,dtype=np.float32).reshape(-1)
          dat_label_weights = compute_weights_array(dat_label)
          #dat_label = convert_importance_to_vector(dat_label)
          print "dat shape: " + str(dat.shape)
          print "dat_label shape: " + str(dat_label.shape)

          t = time.time()
          #frame_importance_numpy = np.asarray(frame_importance[start_count:start_count+10]).reshape(-1)
          _, loss_r, accuracy_r, summary, output, hidden_last_batch_1, hidden_last_batch_2, hidden_last_batch_3, hidden_last_batch_4 = sess.run([train_op, loss, accuracy, tf_tensorboard, output_integer,hidden_feature_map_1,hidden_feature_map_2,hidden_feature_map_3,hidden_feature_map_4],
            feed_dict={x:dat, labels:dat_label,
            hidden_placeholder_1:hidden_input_1,
            hidden_placeholder_2:hidden_input_2,
            hidden_placeholder_3:hidden_input_3,
            hidden_placeholder_4:hidden_input_4, label_weights: dat_label_weights})

          elapsed = time.time() - t

          hidden_input_1 = hidden_last_batch_1
          hidden_input_2 = hidden_last_batch_2
          hidden_input_3 = hidden_last_batch_3
          hidden_input_4 = hidden_last_batch_4


          # Write data to tensorboard
          summary_writer.add_summary(summary, summary_writer_it)
          summary_writer_it += 1

          print "MODEL OUTPUT: " + str(output)
          #print "TRUE OUTPUT: " + str(np.asarray(np.argmax(dat_label,axis=1),dtype=np.int32))
          print "TRUE OUTPUT: " + str(np.asarray(np.round(dat_label),dtype=np.int32))
          
          print "LOSS: " + str(loss_r)
          print "ACCURACY: " + str(accuracy_r)

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










        







