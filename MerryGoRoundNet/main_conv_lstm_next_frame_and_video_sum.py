from __future__ import division
import os.path
import time
from collections import defaultdict
import scipy.misc

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
from PIL import Image


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")

tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")

tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

FPS = 12
BATCH_SIZE = 4
IMAGE_SHAPE = 224
IMAGE_CHANNELS = 1
SEQ_LENGTH = 50
LEARNING_RATE = 0.02
MOMENTUM = 0.9
LABEL_WEIGHT = 1
COLOR_FILTERS = 1
#mean_image = np.zeros((224,224,3))




def get_frame_importance(file_dir):
  f = open(file_dir,"r")
  video_to_frame_importance = dict()
  for video_imp in f:
    tab_separated_values = video_imp.split('\t')
    scores = tab_separated_values[2].split(',')
    i=0
    final_scores = [(float(score)-1) for score in scores[::10]]
    final_scores = [1.0 if score>1.0 else 0.0 for score in final_scores]
    video_to_frame_importance[tab_separated_values[0]]=final_scores
  return video_to_frame_importance

def get_images(frames, mean_image,use_mean_image,convert_to_grey_scale, shape=IMAGE_SHAPE):
  image_rows = shape
  image_columns = shape
  image_channels = 3

  images = [(imread(element)) for element in frames]
  images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,image_channels)
  
  if use_mean_image:
    images = [images[i]-mean_image for i in range(0,SEQ_LENGTH)]
    images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,image_channels)
  

  if convert_to_grey_scale:
    images = np.asarray(images)
    images = np.dot(images[...,:3], [0.299, 0.587, 0.114])
    images = np.asarray(images).reshape(SEQ_LENGTH,shape,shape,1)
  #print "mean image sum: " + str(np.sum(mean_image))

  #print(images.shape)
  return images



def down_sampling_decoders(inputs,hidden_1,hidden_2,scope):
    #14 X 14 X 128

    #DECODER BLOCK 1
    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 256, scope+"_decode_1_1",padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")

    # conv lstm cell 1
    shape = tf.shape(conv1_1)
    with tf.variable_scope(scope+"_conv_lstm_1"):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 256)
      y_1, hidden_feature_map_1, matrix_1 = cell(conv1_1, hidden_1)
      ld.variable_summaries(y_1, scope + "_y_1")
      ld.variable_summaries(hidden_feature_map_1, scope + "_hidden_feature_map[1]")

    ld.variable_summaries(y_1, scope + "_y_1")

    y_1 += conv1_1
    ld.variable_summaries(y_1, scope + "_skip_y_1")



    #7X7X256
    #DECODER BLOCK 2
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(y_1, 3, 2, 512,scope+"_decode_2_1",padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")


    # conv lstm cell 2
    shape = tf.shape(conv2_1)
    with tf.variable_scope(scope+'_conv_lstm_2'):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 512)
      y_2, hidden_feature_map_2, matrix_2 = cell(conv2_1, hidden_2)
      ld.variable_summaries(y_2, scope + "_y_2")
      ld.variable_summaries(hidden_feature_map_2, scope + "_hidden_feature_map[2]")

    ld.variable_summaries(y_2, scope + "_y_2")
    #4 X 4 X 512

    y_2 += conv2_1
    ld.variable_summaries(y_2, scope + "_skip_y_2")

    conv5_1, conv_w5_1, conv_b5_1 = ld.conv_layer(y_2, 3, 2, 1024,scope+"_decode_5_1",padding="SAME")
    ld.variable_summaries(conv5_1, scope + "_conv5_1")

    conv6_1, conv_w6_1, conv_b6_1 = ld.conv_layer(conv5_1, 3, 2, 2048,scope+"_decode_6_1",padding="SAME")
    ld.variable_summaries(conv6_1, scope + "_conv6_1")

    shape = tf.shape(conv6_1)
    y_2 = tf.reshape(conv6_1,[shape[0],1,1,2048])
    # 1 X 1 X 8192
    
    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(y_2, 1, 1, 512,scope+"_decode_3_1",padding="SAME")
    # 1 X 1 X 512
    conv4_1, conv_w4_1, conv_b4_1 = ld.conv_layer(conv3_1, 1, 1, 1,scope+"_decode_4_1",padding="SAME",linear=True)
    # 1 X 1 X 1

    #conv4_1 = tf.nn.sigmoid(conv4_1)

    output = tf.reshape(conv4_1,[-1])

    ld.variable_summaries(conv3_1, scope+ "_conv3_1")
    ld.variable_summaries(output, scope+ "_output")

    return output,hidden_feature_map_1, hidden_feature_map_2



def up_sampling_decoder(inputs,ladder_connection_1,ladder_connection_2,ladder_connection_3,scope):
    #14 X 14 X 128 

    #DECODER BLOCK 1
    shape = tf.shape(inputs)
    output_shape = [shape[0],28,28,64]
    
    conv1_1, conv_w1_1, conv_b1_1 = ld.deconv_layer(inputs, 3, output_shape,"decode_1_1",stride=2,padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")
    #28 X 28 X 64

    conv1_1 += ladder_connection_3
    
    output_shape = [shape[0],56,56,32]
    conv1_2, conv_w1_2, conv_b1_2 = ld.deconv_layer(conv1_1, 3, output_shape,"decode_1_2",stride=2,padding="SAME")
    ld.variable_summaries(conv1_2, scope + "_conv1_2")
    #56 X 56 X 32

    conv1_2 += ladder_connection_2
    ld.variable_summaries(conv1_2, scope + "_skip_conv1_2")


    #DECODER BLOCK 2
    shape = tf.shape(conv1_2)
    output_shape = [shape[0],112,112,16]
    conv2_1, conv_w2_1, conv_b2_1 = ld.deconv_layer(conv1_2, 3, output_shape,"decode_2_1",stride=2,padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")
    #112 X 112 X 16

    conv2_1 += ladder_connection_1

    output_shape = [shape[0],224,224,8]
    conv2_2, conv_w2_2, conv_b2_2 = ld.deconv_layer(conv2_1, 3, output_shape,"decode_2_2",stride=2,padding="SAME")
    ld.variable_summaries(conv2_2, scope + "_conv2_2")
    #224 X 224 X 8

    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(conv2_2, 3, 1, 1,"decode_3_2", padding="SAME",linear=True)
    ld.variable_summaries(conv3_1, scope + "_conv3_2")

    conv3_2 = conv3_1    
    #conv3_2 = tf.nn.sigmoid(conv3_1)

    #224 X 224 X COLOR_FILTERS

    wts_list = []
    feature_map_list = []

    return conv3_2,wts_list,feature_map_list



def network(inputs, hidden_1,hidden_2, summarization_hidden_1,summarization_hidden_2):#,segmentation_hidden_1,segmentation_hidden_2):

    #ENCODER BLOCK 1
    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 16, "encode_1_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_1, "conv1_1")
    conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 32, "encode_1_2",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_2, "conv1_2")

    # conv lstm cell 1
    #56X56X32
    shape = tf.shape(conv1_2)
    with tf.variable_scope('conv_lstm_1'):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 32)
      y_1, hidden_feature_map_1, matrix_1 = cell(conv1_2, hidden_1)
      ld.variable_summaries(y_1, "y_1")
      ld.variable_summaries(hidden_feature_map_1, "hidden_feature_map[1]")

    ld.variable_summaries(y_1, "y_1")
    #56 X 56 X 32

    lstm_1 = y_1
    y_1 += conv1_2
    ld.variable_summaries(y_1, "skip_y_1")


    #ENCODER BLOCK 2
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(y_1, 3, 2, 64,"encode_2_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv2_1, "conv2_1")
    conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 128,"encode_2_2",padding="SAME")
    ld.variable_summaries(conv2_2, "conv2_2")

    # conv lstm cell 2
    #14 X 14X 128
    shape = tf.shape(conv2_2)
    with tf.variable_scope('conv_lstm_2'):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 128)
      y_2, hidden_feature_map_2, matrix_2 = cell(conv2_2, hidden_2)
      ld.variable_summaries(y_2, "y_2")
      ld.variable_summaries(hidden_feature_map_2, "hidden_feature_map[2]")

    ld.variable_summaries(y_2, "y_2")
    #14 X 14 X 128
    

    lstm_2 = y_2
    y_2 += conv2_2
    ld.variable_summaries(y_2, "skip_y_2")

    with tf.variable_scope("prednet"):
        prednet_output,prednet_wts_list,prednet_feature_map_list = up_sampling_decoder(y_2,conv1_1,y_1,conv2_1,"prednet")

    with tf.variable_scope("frame_importance"):
        summarization_output,summarization_hidden_feature_map_1,summarization_hidden_feature_map_2 = down_sampling_decoders(y_2,summarization_hidden_1,summarization_hidden_2,"frame_importance")


    wts_list = [conv_w1_1, conv_b1_1, conv_w1_2, conv_b1_2, conv_w2_1, conv_b2_1,
    conv_w2_2, conv_b2_2, matrix_1, matrix_2]

    feature_map_list = [conv1_1,conv1_2,conv2_1,conv2_2,y_1,y_2,lstm_1,lstm_2]

    return prednet_output,hidden_feature_map_1, hidden_feature_map_2,wts_list,feature_map_list,prednet_wts_list,prednet_feature_map_list,summarization_output,summarization_hidden_feature_map_1,summarization_hidden_feature_map_2


# make a template for reuse
network_template = tf.make_template('network', network)



def train():
  """Train merry_go_round_net"""

  with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, 3])
    labels = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE])
    keep_prednet_loss = tf.placeholder(tf.int32 , [None, SEQ_LENGTH])
    summarization_labels = tf.placeholder(tf.float32, [None])
    
    
    stddev = 0.2
    hidden_placeholder_1 = tf.placeholder(tf.float32,[BATCH_SIZE,56,56,64])
    hidden_placeholder_2 = tf.placeholder(tf.float32,[BATCH_SIZE,14,14,256])
    summarization_hidden_placeholder_1 = tf.placeholder(tf.float32,[BATCH_SIZE,7,7,512])
    summarization_hidden_placeholder_2 = tf.placeholder(tf.float32,[BATCH_SIZE,4,4,1024])

    x_dropout = x

    x_unwrap = []
    summarization_x_unwrap = []

    with tf.variable_scope("MerryGoRoundNet") as scope_:
        device_count = 0
        with tf.device("/gpu:" + str(device_count)):
            x_1, hidden_feature_map_1,hidden_feature_map_2,_,_,_,_,summarization_output,summarization_hidden_feature_map_1,summarization_hidden_feature_map_2 = network_template(x_dropout[:,0,:,:,:], 
              hidden_1 = hidden_placeholder_1,
              hidden_2 = hidden_placeholder_2,
              summarization_hidden_1 = summarization_hidden_placeholder_1,
              summarization_hidden_2 = summarization_hidden_placeholder_2)
            x_unwrap.append(x_1)
            summarization_x_unwrap.append(summarization_output)
            
        scope_.reuse_variables()
        gpu_devices = [i for i in range(0,8)]
        for i in xrange(1,SEQ_LENGTH):
            with tf.device("/gpu:" + str(device_count)):
              x_1, hidden_feature_map_1,hidden_feature_map_2, wts_list, feature_map_list, prednet_wts_list, prednet_feature_map_list,summarization_output,summarization_hidden_feature_map_1,summarization_hidden_feature_map_2 = network_template(x_dropout[:,i,:,:,:], 
                hidden_1 = hidden_feature_map_1,
                hidden_2 = hidden_feature_map_2,
                summarization_hidden_1 = summarization_hidden_feature_map_1,
                summarization_hidden_2 = summarization_hidden_feature_map_2)
              x_unwrap.append(x_1)
              summarization_x_unwrap.append(summarization_output)
            device_count+=1
            device_count%=1

    input_image_to_print = tf.reshape(labels,[-1,IMAGE_SHAPE, IMAGE_SHAPE, 1])
    image_to_print = tf.nn.sigmoid(x_unwrap)
    image_to_print = tf.reshape(image_to_print,[-1,IMAGE_SHAPE,IMAGE_SHAPE,1])
    tf.summary.image('output_image',image_to_print)
    tf.summary.image('input_image',input_image_to_print)
    
    
    x_unwrap = tf.reshape(x_unwrap,[-1,SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE])
    print("X_UNWRAP SHAPE: " + str(x_unwrap.shape))
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_unwrap,labels=labels)
    #loss = tf.square(x_unwrap-labels)
    reshaped_loss = tf.reshape(loss,[-1,SEQ_LENGTH,IMAGE_SHAPE*IMAGE_SHAPE*1])
    reduced_loss = tf.reduce_mean(reshaped_loss,axis=2)
    print " LOSS SHAPE: "  + str(loss.shape)
    clean_nf_loss = reduced_loss * tf.cast(keep_prednet_loss, tf.float32)
    clean_nf_loss = tf.reshape(clean_nf_loss,[-1])
    loss = tf.reduce_mean(clean_nf_loss)

    summarization_output = tf.reshape(summarization_x_unwrap,[-1])
    sigmoid_summarization_output = tf.nn.sigmoid(summarization_output)
    #summarization_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=summarization_output,labels=summarization_labels)
    summarization_loss = tf.square(sigmoid_summarization_output - summarization_labels)
    summarization_loss = tf.reduce_mean(summarization_loss)
    
    reduced_recall_value = np.asarray(1-clean_nf_loss) * np.asarray(sigmoid_summarization_output)
    reduced_recall_loss = tf.reduce_mean(reduced_recall_value)


    final_loss = summarization_loss + loss  + 10*reduced_recall_loss


    
    grad_list = tf.gradients(loss, wts_list)
    grad_feature_map_list = tf.gradients(loss, feature_map_list)

    wts_list_var_name = ["conv_w1_1", "conv_b1_1", "conv_w1_2", "conv_b1_2", "conv_w2_1", "conv_b2_1",
    "conv_w2_2", "conv_b2_2", "matrix_1", "matrix_2"]

    feature_map_list_name = ["conv1_1","conv1_2","conv2_1","conv2_2","y_1","y_2","lstm_1","lstm_2"]


    for grad_num in range(0, len(wts_list_var_name)):
      ld.variable_summaries(grad_list[grad_num], "grad_" + wts_list_var_name[grad_num])
      
    for grad_num in range(0,len(feature_map_list_name)):
      ld.variable_summaries(grad_feature_map_list[grad_num], "grad_"+feature_map_list_name[grad_num])
    

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('final_loss', final_loss)
    tf.summary.scalar('summarization_loss', summarization_loss)

    # training
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, colocate_gradients_with_ops=True)
    #train_op = tf.train.MomentumOptimizer(LEARNING_RATE,MOMENTUM).minimize(loss, colocate_gradients_with_ops=True)
    

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
    videos = glob.glob("../dataset/video_frames/Webscope_I4/[!.]*")
    labels_map = get_frame_importance("../dataset/Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv")

    print(len(videos))

    video_object_detection_frame_directory = "../dataset/video_frames/Webscope_I4_boundary_detection/"
    video_frame_boundary_map = defaultdict(lambda:defaultdict(lambda:0))
    for video_folder in os.listdir(video_object_detection_frame_directory):
        if video_folder[0]=='.':
            continue
    for frame_number in os.listdir(video_object_detection_frame_directory + video_folder):
        #frame number would contain ".jpg"
        video_frame_boundary_map[video_folder][frame_number] = 1

    
    saver_step = 0

    mean_image = np.load("mean_image.npy")
      
    counter = 0
    while(epoch<MAX_EPOCHS):
        print("Epoch: " + str(epoch))
        random.shuffle(videos)
        print "Len of videos glob: " + str(len(videos))
        start_count = 0
        epoch+=1

        #loading the batches
        while start_count+BATCH_SIZE<=len(videos):
            #print "Loading the batches"
            counter+=1
            print("counter: " + str(counter))
            videos_in_iteration = videos[start_count: start_count+BATCH_SIZE]
            start_count += BATCH_SIZE
            video_index = 0
            frame = dict()
            mini_len = 1000000
        
            video_index_video_mapping = defaultdict(lambda:0)
            this_frame_importance = defaultdict(lambda:0)
            #Load glob of each video in mini batch
            for video in videos_in_iteration:
                #print "Load glob of each video in mini batch"
                frame[video_index] = glob.glob(video+"/*.jpg")
                video_index_video_mapping[video_index] = video.split('/')[-1]
                this_frame_importance[video_index] = labels_map[video.split('/')[-1]]
          
                mini_len = min(mini_len, len(frame[video_index]))
                #max 1000 frames selected per video
                mini_len = min(mini_len,1000)
                video_index += 1

            #Operating on glob of each individual video
            frame_start_count = 0

            hidden_input_1 = np.zeros((BATCH_SIZE,56,56,64),dtype=np.float32)
            hidden_input_2 = np.zeros((BATCH_SIZE,14,14,256),dtype=np.float32)

            summarization_hidden_input_1 = np.zeros((BATCH_SIZE,7,7,512),dtype=np.float32)
            summarization_hidden_input_2 = np.zeros((BATCH_SIZE,4,4,1024),dtype=np.float32)

        
 
            while frame_start_count+SEQ_LENGTH+1<=mini_len:
                keep_prednet_array_batch = []
                selected_batch = []
                output_selected_batch = []
                selected_importance_labels = []
                scene_detection_labels = []
                for video_index_num in range(0, video_index):
                    selected_frames = frame[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH]
                    output_selected_frames = frame[video_index_num][frame_start_count+1: frame_start_count+SEQ_LENGTH+1]
                    selected_importance_labels.append(this_frame_importance[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH])
                    keep_prednet_array = [0 if video_frame_boundary_map[video_index_video_mapping[video_index_num]][frame_number.split('/')[-1]]==1 else 1 for frame_number in output_selected_frames]
                    keep_prednet_array[0] = 0
                    selected_batch.append(get_images(selected_frames,mean_image,1,0))
                    output_selected_batch.append(get_images(output_selected_frames,mean_image,0,1)/255.0)
                    keep_prednet_array_batch.append(keep_prednet_array)

                frame_start_count += SEQ_LENGTH
                input_data = np.asarray(selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,3)
                prednet_output_data = np.asarray(output_selected_batch).reshape(BATCH_SIZE,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE)
                keep_prednet_array_batch = np.asarray(keep_prednet_array_batch).reshape([BATCH_SIZE,SEQ_LENGTH])
                dat_label = np.asarray(selected_importance_labels,dtype=np.int32).reshape(-1)

           

                t = time.time()
                #frame_importance_numpy = np.asarray(frame_importance[start_count:start_count+10]).reshape(-1)
                _, loss_r,loss_s,loss_f, loss_rr,summary, hidden_last_batch_1, hidden_last_batch_2,summarization_hidden_last_batch_1, summarization_hidden_last_batch_2,summarization_output_,rr_value = sess.run([train_op, loss,summarization_loss,final_loss,reduced_recall_loss,tf_tensorboard,hidden_feature_map_1,hidden_feature_map_2,summarization_hidden_feature_map_1,summarization_hidden_feature_map_2,sigmoid_summarization_output,reduced_recall_value],
                feed_dict={x:input_data, 
                labels:prednet_output_data,
                keep_prednet_loss:keep_prednet_array_batch, 
                summarization_labels:dat_label,
                hidden_placeholder_1:hidden_input_1,
                hidden_placeholder_2:hidden_input_2,
                summarization_hidden_placeholder_1:summarization_hidden_input_1,
                summarization_hidden_placeholder_2:summarization_hidden_input_2})

                elapsed = time.time() - t

                hidden_input_1 = hidden_last_batch_1
                hidden_input_2 = hidden_last_batch_2
                summarization_hidden_input_1 = summarization_hidden_last_batch_1
                summarization_hidden_input_2 = summarization_hidden_last_batch_2


                # Write data to tensorboard
                summary_writer.add_summary(summary, summary_writer_it)
                summary_writer_it += 1


                print "NF LOSS: " + str(loss_r)
                print "SM LOSS: " + str(loss_s)
                print "RR LOSS: " + str(loss_rr)
                print "FINAL LOSS: " + str(loss_f)
                print("Summarization output  :  " + str(np.rint(np.asarray(summarization_output_)+0.01)))
                print("True output  :  " + str(dat_label))
                print("rr_value: " + str(np.asarray(rr_value)))


                print("Iteration: " + str(saver_step))
                saver_step+=1

                      
                if saver_step%50 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=saver_step)  
                    print("saved to " + FLAGS.train_dir)


            if counter%4==5:    
                # hidden_input_1 = np.zeros((BATCH_SIZE,56,56,64),dtype=np.float32)
                # hidden_input_2 = np.zeros((BATCH_SIZE,14,14,256),dtype=np.float32)
                # summarization_hidden_input_1 = np.zeros((BATCH_SIZE,7,7,512),dtype=np.float32)
                # summarization_hidden_input_2 = np.zeros((BATCH_SIZE,4,4,1024),dtype=np.float32)


                selected_batch = []
                output_selected_batch = []
                keep_prednet_array_batch = []
                selected_frames = frame[video_index-1][0:SEQ_LENGTH]
                output_selected_frames = frame[video_index-1][1: SEQ_LENGTH+1]
                
                selected_importance_labels = []
                selected_importance_labels.append(this_frame_importance[video_index_num][frame_start_count: frame_start_count+SEQ_LENGTH])               
                dat_label = np.asarray(selected_importance_labels,dtype=np.int32).reshape(-1)
                                
                selected_batch.append(get_images(selected_frames,mean_image,1,0))
                output_selected_batch.append(get_images(output_selected_frames,mean_image,0,1)/255.0)
                
                input_data = np.asarray(selected_batch).reshape(1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,3)
                prednet_output_data = np.asarray(output_selected_batch).reshape(1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE)
                
                keep_prednet_array = [0 if video_frame_boundary_map[video_index_video_mapping[video_index-1]][frame_number.split('/')[-1]]==1 else 1 for frame_number in output_selected_frames]
                keep_prednet_array[0] = 0
                                    
                keep_prednet_array_batch.append(keep_prednet_array)
                keep_prednet_array_batch = np.asarray(keep_prednet_array_batch).reshape([1,SEQ_LENGTH])
                                    

                output_images = x_unwrap.eval(session=sess,feed_dict={x:input_data, labels:prednet_output_data,keep_prednet_loss:keep_prednet_array_batch, summarization_labels:dat_label
                # hidden_placeholder_1:hidden_input_1,
                # hidden_placeholder_2:hidden_input_2,
                # summarization_hidden_placeholder_1:summarization_hidden_input_1,
                # summarization_hidden_placeholder_2:summarization_hidden_input_2
                })

                output_images = np.array(output_images).reshape(SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE)
                if not os.path.isdir('PREDNET_OUTPUT'):
                    os.system('mkdir ' + 'PREDNET_OUTPUT')
                for i in range(SEQ_LENGTH):
                    scipy.misc.imsave('PREDNET_OUTPUT/outfile_' + str(i) + '.jpg', output_images[i])





def main(argv=None):  # pylint: disable=unused-argument
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
