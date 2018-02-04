
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
print sys.path
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")


tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")


tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

FPS = 3
BATCH_SIZE = 4
IMAGE_SHAPE = 224
IMAGE_CHANNELS = 3
SEQ_LENGTH = 10
LEARNING_RATE = 0.001

def get_frame_importance(file_dir):
  f = open(file_dir,"r")
  video_to_frame_importance = dict()
  for video_imp in f:
    tab_separated_values = video_imp.split('\t')
    scores = tab_separated_values[2].split(',')
    i=0
    final_scores = [int(score)-1 for score in scores[::10]]
    video_to_frame_importance[tab_separated_values[0]]=final_scores
  return video_to_frame_importance

def get_images(frames,shape=IMAGE_SHAPE):
  image_rows = shape
  image_columns = shape
  image_channels = IMAGE_CHANNELS

  images = [(imread(element)) for element in frames]

  images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,image_channels)
  print images.shape
  return images


def network(inputs, hidden, lstm=True):


  conv1_1 = ld.conv_layer(inputs, 3, 1, 16, "encode_1_1")
  conv1_2 = tf.nn.max_pool(ld.conv_layer(conv1_1, 3, 1, 16, "encode_1_2"), [1,3,3,1],strides=[1,2,2,1], padding="SAME")


  # conv2
  conv2_1 = ld.conv_layer(conv1_2, 3, 1, 32, "encode_2_1")
  conv2_2 = tf.nn.max_pool(ld.conv_layer(conv2_1, 3, 1, 32, "encode_2_2"), [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  # 32 x 32 x 128


  # conv lstm cell 1
  shape = tf.shape(conv2_2)
  with tf.variable_scope('conv_lstm_1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
    if hidden[0] is None:
      hidden[0] = cell.zero_state(BATCH_SIZE, tf.float32) 
    y_1, hidden[0] = cell(conv2_2, hidden[0])


  # conv lstm cell 2
  shape = tf.shape(y_1)
  with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
    if hidden[1] is None:
      hidden[1] = cell.zero_state(BATCH_SIZE, tf.float32) 
    y_2, hidden[1] = cell(y_1, hidden[1])

  y_2_pool = tf.nn.max_pool(y_2, [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  #16 x 16 x 64

  # conv3
  conv3_1 = ld.conv_layer(y_2_pool, 3, 1, 128, "encode_3_1")
  conv3_2 = tf.nn.max_pool(ld.conv_layer(conv3_1, 3, 1, 128, "encode_3_2"), [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  #8 x 8 x 128

  # conv4
  conv4_1 = ld.conv_layer(conv3_2, 3, 1, 256, "encode_4_1")
  conv4_2 = tf.nn.max_pool(ld.conv_layer(conv4_1, 3, 1, 256, "encode_4_2"), [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  conv4_3 = tf.nn.max_pool(ld.conv_layer(conv4_2, 3, 1, 256, "encode_4_3"), [1,3,3,1],strides=[1,2,2,1], padding="SAME")
  #4 x 4 x 256

  
  fn1 = ld.fc_layer(conv4_3, 1024, flat=True,idx="fc_1")
  fn2 = ld.fc_layer(fn1, 1024,idx="fc_2")
  output = (ld.fc_layer(fn2, 5, linear = True,idx="fc_3"))

  return output, hidden




# make a template for reuse
network_template = tf.make_template('network', network)




def train():
  """Train ring_net for a number of steps."""


  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])

    labels = tf.placeholder(tf.int64, [BATCH_SIZE*SEQ_LENGTH])
    # possible dropout inside
    x_dropout = x

    # create network
    x_unwrap = []

    # conv network
    hidden = [None for i in range(2)]

    gpu_devices = [i for i in range(0,8)]
    device_count = 0
    for i in xrange(SEQ_LENGTH):
        with tf.device("/gpu:" + str(device_count)):
          x_1, hidden = network_template(x_dropout[:,i,:,:,:], hidden)
          x_unwrap.append(x_1)
        device_count+=1
        device_count%=8



    #SHAPE OF X_WRAP : BATCH_SIZE * SEQ_LENGTH, 5
    x_unwrap = tf.reshape(x_unwrap,[-1, 5])

    print "LABELS SHAPE: " + str(labels)
    print "X_UNWRAP SHAPE: " + str(x_unwrap)
    correct_prediction = tf.equal(tf.argmax(x_unwrap, axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    


    sigmoid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x_unwrap)
    print "LOSS SHAPE: "  + str(sigmoid_loss.shape)
    loss = tf.reduce_sum(sigmoid_loss)/BATCH_SIZE
    print "LOSS SHAPE: "  + str(loss.shape)
    
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, colocate_gradients_with_ops=True)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph
    tf_tensorboard = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph_def)
    summary_writer_it = 0
    saver = tf.train.Saver()

    MAX_EPOCHS = 10000000
    epoch = 0
    videos = glob.glob("../dataset/video_frames/Webscope_I4/[!.]*")

    labels_map = get_frame_importance("../dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv")
    #loading the epochs
    saver_step = 0

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

        while frame_start_count+SEQ_LENGTH<mini_len:
          selected_batch = []
          selected_importance_labels = []
          for video_index_num in range(0, video_index):
            selected_frames = frame[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH]
            selected_batch.append(get_images(selected_frames))
            selected_importance_labels.append(this_frame_importance[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH])

          frame_start_count += SEQ_LENGTH
          dat = np.asarray(selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
          dat_label = np.asarray(selected_importance_labels).reshape(-1)
          print "dat shape: " + str(dat.shape)
          print "dat_label shape: " + str(dat_label.shape)
          print "dat label: " + str(dat_label)

          t = time.time()
          #frame_importance_numpy = np.asarray(frame_importance[start_count:start_count+10]).reshape(-1)
          _, loss_r, accuracy_r, summary = sess.run([train_op, loss, accuracy, tf_tensorboard],feed_dict={x:dat, labels:dat_label})
          elapsed = time.time() - t


          # Write data to tensorboard
          summary_writer.add_summary(summary, summary_writer_it)
          summary_writer_it += 1

          print "LOSS: " + str(loss_r)
          print "ACCURACY: " + str(accuracy_r)

          print("Iteration: " + str(saver_step))
          saver_step+=1

                  
          if saver_step%5 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=saver_step)  
            print("saved to " + FLAGS.train_dir)
            





def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()





if __name__ == '__main__':
  tf.app.run()










        







