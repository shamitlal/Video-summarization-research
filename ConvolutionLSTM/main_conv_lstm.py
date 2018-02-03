
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import layer_def as ld
import BasicConvLSTMCell
import glob
from scipy.misc import imread, imsave, imresize
import sys
sys.path.append(os.path.abspath('..'))
print sys.path
import utils
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 


'''
def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 
'''

def get_images(frames,shape):
  image_rows = shape
  image_columns = shape
  image_channels = 3

  for frame in frames:
    images = [imresize(np.asarray(Image.open(element)), (image_rows,image_columns,image_channels)) for element in frames]

  images = np.asarray(images).reshape(1,10,shape,shape,3)
  print images.shape
  return images


def network(inputs, hidden, lstm=True):
  conv1 = tf.nn.max_pool(ld.conv_layer(inputs, 3, 1, 16, "encode_1"), [1,2,2,1],strides=[1,1,1,1], padding="VALID")
  # conv2
  conv2 = tf.nn.max_pool(ld.conv_layer(conv1, 3, 1, 16, "encode_2"), [1,2,2,1],strides=[1,1,1,1], padding="VALID")
  # conv3
  #conv3 = ld.conv_layer(conv2, 3, 1, 16, "encode_3")
  conv3 = conv2
  # conv4
  conv4 = ld.conv_layer(conv3, 3, 1, 32, "encode_4")
  shape = tf.shape(conv4)
  print "shape: " + str(shape)
  y_0 = conv4
  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
      if hidden[0] is None:
        hidden[0] = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_1, hidden[0] = cell(y_0, hidden[0])
  

  shape = tf.shape(y_1)
  print "shape after first conv lstm layer: " + str(shape)

  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
      if hidden[1] is None:
        hidden[1] = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_2, hidden[1] = cell(y_1, hidden[1])
 
  
  # conv5
  conv5 = tf.nn.max_pool(ld.conv_layer(conv1, 3, 1, 64, "encode_5"), [1,2,2,1],strides=[1,1,1,1], padding="VALID")
  # conv5
  conv6 = tf.nn.max_pool(ld.conv_layer(conv1, 3, 1, 64, "encode_6"), [1,2,2,1],strides=[1,1,1,1], padding="VALID")
  # fully connected 1
  print "shape of conv6: " + str(tf.shape(conv6))
  fn1 = ld.fc_layer(conv6, 1024, flat=True,idx="fc_1")

  fn2 = ld.fc_layer(fn1, 512,idx="fc_2")

  fn3 = ld.fc_layer(fn2, 128,idx="fc_3")

  output = (ld.fc_layer(fn3, 1, linear = True,idx="fc_4"))

  return output, hidden




# make a template for reuse
network_template = tf.make_template('network', network)




def train():
  """Train ring_net for a number of steps."""


  with tf.Graph().as_default():
    SEQ_LENGTH = 10
    # make inputs
    x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, 128, 128, 3])

    labels = tf.placeholder(tf.float32, [SEQ_LENGTH])
    # possible dropout inside
    x_dropout = x

    # create network
    x_unwrap = []

    # conv network
    hidden = [None for i in range(2)]

    for i in xrange(SEQ_LENGTH):
        x_1, hidden = network_template(x_dropout[:,i,:,:,:], hidden)
        x_unwrap.append(x_1)


    #SHAPE OF X_WRAP : BATCH_SIZE * SEQ_LENGTH
    x_unwrap = tf.reshape(x_unwrap,[-1])

    print "LABELS SHAPE: " + str(labels)
    print "X_UNWRAP SHAPE: " + str(labels)
    
    


    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=x_unwrap)
    print "LOSS SHAPE: "  + str(sigmoid_loss.shape)
    loss = tf.reduce_sum(sigmoid_loss)
    print "LOSS SHAPE: "  + str(loss.shape)
    
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
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
    frame_importance = utils.get_frame_importance_vector("../dataset/true_summaries/video4",367)
    print "FRAME_IMPORTANCE: "  + str(frame_importance)
    frames = glob.glob("../dataset/video_frames/video4/" + "*.jpg")
    print "FRAMES GLOB: " + str(frames)
    

    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph
    tf_tensorboard = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph_def)
    summary_writer_it = 0

    MAX_EPOCHS = 10000000
    epoch = 0
    while(epoch<MAX_EPOCHS):
      MAX_STEP = 360/10
      start_count = 0
      for step in xrange(MAX_STEP):
        #dat = form_data(FLAGS.batch_size, FLAGS.seq_length, 128)
        dat = get_images(frames[start_count:start_count+10],128)
        print "IMAGES SHAPE: " + str(dat.shape)

        t = time.time()
        frame_importance_numpy = np.asarray(frame_importance[start_count:start_count+10]).reshape(-1)
        _, loss_r, summary = sess.run([train_op, loss, tf_tensorboard],feed_dict={x:dat, labels:frame_importance_numpy})
        elapsed = time.time() - t


        # Write data to tensorboard
        summary_writer.add_summary(summary, summary_writer_it)
        summary_writer_it += 1

        print "LOSS: " + str(loss_r)

        start_count += 10






      '''
      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        # make video
        print("now generating video!")
        video = cv2.VideoWriter()
        success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
        dat_gif = dat
        ims = sess.run([x_unwrap_g],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
        ims = ims[0][0]
        print(ims.shape)
        for i in xrange(50 - FLAGS.seq_start):
          x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
          new_im = cv2.resize(x_1_r, (180,180))
          video.write(new_im)
        video.release()
        '''





def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()





if __name__ == '__main__':
  tf.app.run()


