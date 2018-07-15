import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import layer_def as ld
import BasicConvLSTMCell
import scipy.misc

LEARNING_RATE = 0.005
BATCH_SIZE = 1
IMAGE_SHAPE = 64
IMAGE_CHANNELS = 1
SEQ_LENGTH = 19


    

def encoder(inputs,hidden_1,hidden_2):
    scope = 'encoder'
    #64X64X1

    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 16, "encode_1_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_1, "conv1_1")

    #32X32X16
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(conv1_1, 3, 2, 32,"encode_2_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv2_1, "conv2_1")
    
    #16X16X32
    shape = tf.shape(conv2_1)
    with tf.variable_scope('conv_lstm_1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 32)
      y_1, hidden_feature_map_1, matrix_1 = cell(conv2_1, hidden_1,scope="lstm1")
      ld.variable_summaries(y_1, "y_1")
      ld.variable_summaries(hidden_feature_map_1, "hidden_feature_map[1]")


    
    lstm_1 = y_1
    #y_1 += conv2_1
    ld.variable_summaries(y_1, "skip_y_1")


    #16X16X32
    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(y_1, 3, 2, 64,"encode_3_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv3_1, "conv3_1")
    
    #8X8X64
    shape = tf.shape(conv3_1)
    with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([shape[1], shape[2]], [3,3], 64)
      y_2, hidden_feature_map_2, matrix_2 = cell(conv3_1, hidden_2,scope="lstm2")
      ld.variable_summaries(y_2, "y_2")
      ld.variable_summaries(hidden_feature_map_2, "hidden_feature_map[2]")

    ld.variable_summaries(y_2, "y_2")
    
    lstm_2 = y_2
    #y_2 += conv3_1
    ld.variable_summaries(y_2, "skip_y_2")


    wts_list = [conv_w1_1, conv_b1_1, conv_w2_1, conv_b2_1,conv_w3_1, conv_b3_1,matrix_1,matrix_2]
    feature_map_list = [conv1_1,conv2_1,conv3_1,y_1,y_2,lstm_1,lstm_2]

    #8X8X64
    output,decoder_wts_list,decoder_feature_map_list = decoder(y_2,y_1)
    
    return output,hidden_feature_map_1,hidden_feature_map_2,wts_list,feature_map_list,decoder_wts_list,decoder_feature_map_list


def decoder(inputs,y_1):
    scope = 'decoder'

    #8X8X64
    shape = tf.shape(inputs)
    output_shape = [shape[0],16,16,32]
    conv1_1, conv_w1_1, conv_b1_1 = ld.deconv_layer(inputs, 3, output_shape,"decode_1_1",stride=2,padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")

    conv1_1 += y_1
    skip_conv = conv1_1
    ld.variable_summaries(skip_conv, scope + "skip_conv")
    
    #16X16X32
    shape = tf.shape(conv1_1)
    output_shape = [shape[0],32,32,16]
    conv2_1, conv_w2_1, conv_b2_1 = ld.deconv_layer(conv1_1, 3, output_shape,"decode_2_1",stride=2,padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")

    #32X32X16
    shape = tf.shape(conv2_1)
    output_shape = [shape[0],64,64,8]    
    conv3_1, conv_w3_1, conv_b3_1 = ld.deconv_layer(conv2_1, 3, output_shape,"decode_3_1",stride=2,padding="SAME")
    ld.variable_summaries(conv3_1, scope + "_conv3_1")
    
    #64X64X8
    conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 1, 1,"decode_3_2", padding="SAME",linear=True)
    
    conv3_2  = tf.nn.sigmoid(conv3_2)
    ld.variable_summaries(conv3_2, scope + "_conv3_2")

    wts_list = [conv_w1_1, conv_b1_1, conv_w2_1, conv_b2_1,conv_w3_1, conv_b3_1, conv_w3_2, conv_b3_2]
    feature_map_list = [conv1_1,conv2_1,conv3_1,conv3_2]

    #64X64X1
    return conv3_2,wts_list,feature_map_list


def train():
    x = tf.placeholder(tf.float32,[None,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS])
    next_frame = tf.placeholder(tf.float32, [None, SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE,IMAGE_CHANNELS])
    #keep_prednet_loss = tf.placeholder(tf.int32 , [None, SEQ_LENGTH])
    hidden_placeholder_1 = tf.placeholder(tf.float32,[BATCH_SIZE,16,16,64])
    hidden_placeholder_2 = tf.placeholder(tf.float32,[BATCH_SIZE,8,8,128])

    with tf.variable_scope("victory") as scope_:
        x_unwrap = []
        device_count = 0
        with tf.device("/gpu:" + str(device_count)):
            x_1,hidden_feature_map_1,hidden_feature_map_2,_,_,_,_ = encoder(x[:,0,:,:,:], 
              hidden_1 = hidden_placeholder_1,
              hidden_2 = hidden_placeholder_2)
            x_unwrap.append(x_1)

        scope_.reuse_variables()
        gpu_devices = [i for i in range(0,8)]
        for i in xrange(1,SEQ_LENGTH):
            with tf.device("/gpu:" + str(device_count)):
              x_1, hidden_feature_map_1,hidden_feature_map_2,wts_list, feature_map_list,decoder_wts_list,decoder_feature_map_list = encoder(x[:,i,:,:,:], 
                hidden_1 = hidden_feature_map_1,
                hidden_2 = hidden_feature_map_2)
              x_unwrap.append(x_1)
            device_count+=1
            device_count%=1



    x_unwrap = tf.reshape(x_unwrap,[-1,SEQ_LENGTH, IMAGE_SHAPE, IMAGE_SHAPE, IMAGE_CHANNELS])
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_unwrap,labels=next_frame)
    #loss  = tf.square(output_image_sigmoid-x)
    loss = tf.reduce_mean(loss)

    tf.summary.image('output_image',tf.reshape(x_unwrap,[-1,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS]))
    tf.summary.image('input_image',tf.reshape(next_frame,[-1,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS]))
    tf.summary.scalar('loss', loss)

    grad_list = tf.gradients(loss, wts_list)
    grad_feature_map_list = tf.gradients(loss, feature_map_list)
    wts_list_var_name = ["conv_w1_1","conv_b1_1", "conv_w2_1", "conv_b2_1","conv_w3_1", "conv_b3_1", "matrix_1", "matrix_2"]
    feature_map_list_name = ["conv1_1","conv2_1","conv3_1","y_1","y_2","lstm_1","lstm_2"]

    for grad_num in range(0, len(wts_list_var_name)):
      ld.variable_summaries(grad_list[grad_num], "grad_" + wts_list_var_name[grad_num])
      
    for grad_num in range(0,len(feature_map_list_name)):
      ld.variable_summaries(grad_feature_map_list[grad_num], "grad_"+feature_map_list_name[grad_num])


    decoder_grad_list = tf.gradients(loss, decoder_wts_list)
    decoder_grad_feature_map_list = tf.gradients(loss, decoder_feature_map_list)
    decoder_wts_list_var_name = ["conv_w1_1", "conv_b1_1", "conv_w2_1", "conv_b2_1","conv_w3_1", "conv_b3_1", "conv_w3_2", "conv_b3_2"]
    decoder_feature_map_list_name = ["conv1_1","conv2_1","conv3_1","conv3_2"]

    for grad_num in range(0, len(decoder_wts_list_var_name)):
      ld.variable_summaries(decoder_grad_list[grad_num], "grad_" + decoder_wts_list_var_name[grad_num])
      
    for grad_num in range(0,len(decoder_feature_map_list_name)):
      ld.variable_summaries(decoder_grad_feature_map_list[grad_num], "grad_"+decoder_feature_map_list_name[grad_num])



    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, colocate_gradients_with_ops=True)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    graph_def = sess.graph
    tf_tensorboard = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./checkpoints/train_store_conv_lstm",graph_def)
    summary_writer_it = 0


    print('Training...')
    global mnsit_data
    mnsit_data_original = np.load('../dataset/mnist_test_seq.npy')
    mnsit_data = mnsit_data_original.swapaxes(0,1)
    
    for i in range(10001):
        start_count = 0
        
        batch_count = 0
        while start_count+BATCH_SIZE < 10000:
            batch_count +=1
            selected_frames_batch = []
            next_selected_frames_batch = []
            hidden_input_1 = np.zeros((BATCH_SIZE,16,16,64),dtype=np.float32)
            hidden_input_2 = np.zeros((BATCH_SIZE,8,8,128),dtype=np.float32)
            
            for video_index in range(start_count,start_count+BATCH_SIZE):
                selected_frames = mnsit_data[video_index][:-1]/255.0
                selected_frames = np.array(selected_frames).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                selected_frames_batch.append(selected_frames)

                next_selected_frames = mnsit_data[video_index][1:]/255.0
                next_selected_frames = np.array(selected_frames).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                next_selected_frames_batch.append(selected_frames)


            selected_frames_batch = np.array(selected_frames_batch).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
            next_selected_frames_batch = np.array(next_selected_frames_batch).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
            
            _,generated_loss,summary = sess.run([train_op,loss,tf_tensorboard],feed_dict={x: selected_frames_batch,
                next_frame:next_selected_frames_batch, 
                hidden_placeholder_1:hidden_input_1,
                hidden_placeholder_2:hidden_input_2})
        
            print("LOSS: " + str(generated_loss))

            summary_writer.add_summary(summary, summary_writer_it)
            summary_writer_it += 1
            

            if batch_count%200==0:
                selected_frames_batch = []
                next_selected_frames_batch = []
                selected_frames = mnsit_data[start_count][:-1]/255.0
                selected_frames = np.array(selected_frames).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                selected_frames_batch.append(selected_frames)

                next_selected_frames = mnsit_data[start_count][1:]/255.0
                next_selected_frames = np.array(selected_frames).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                next_selected_frames_batch.append(selected_frames)


                selected_frames_batch = np.array(selected_frames_batch).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                next_selected_frames_batch = np.array(next_selected_frames_batch).reshape(-1,SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
            
                output_images = x_unwrap.eval(session=sess,feed_dict={x: selected_frames_batch,
                next_frame:next_selected_frames_batch, 
                hidden_placeholder_1:hidden_input_1,
                hidden_placeholder_2:hidden_input_2})

                output_images = np.array(output_images).reshape(SEQ_LENGTH,IMAGE_SHAPE,IMAGE_SHAPE)
                if not os.path.isdir('MOVING_MNIST_OUTPUT'):
                    os.system('mkdir ' + 'MOVING_MNIST_OUTPUT')
                for i in range(SEQ_LENGTH):
                    scipy.misc.imsave('MOVING_MNIST_OUTPUT/outfile_' + str(i) + '.jpg', output_images[i])


            start_count += BATCH_SIZE

if __name__ == "__main__":
    train()
