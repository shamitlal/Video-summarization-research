import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import layer_def as ld
import BasicConvLSTMCell

LEARNING_RATE = 0.005
BATCH_SIZE = 6
IMAGE_SHAPE = 64
IMAGE_CHANNELS = 1

def encoder(inputs):
    scope = 'encoder'
    #inputs = tf.reshape(inputs,[-1,28,28,1])

    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 16, "encode_1_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_1, "conv1_1")
    conv1_2 =conv1_1
    #conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 32, "encode_1_2",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_2, "conv1_2")

    #32X32X16
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(conv1_2, 3, 2, 32,"encode_2_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv2_1, "conv2_1")
    conv2_2 = conv2_1
    #conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 64,"encode_2_2",padding="SAME")
    ld.variable_summaries(conv2_2, "conv2_2")

    #16X16X32
    conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(conv2_2, 3, 2, 64,"encode_3_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv3_1, "conv3_1")
    conv3_2 = conv3_1
    #conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 1, 32,"encode_3_2", padding="SAME")
    ld.variable_summaries(conv3_2, "conv3_2")

    wts_list = [conv_w1_1, conv_b1_1, conv_w2_1, conv_b2_1,conv_w3_1, conv_b3_1]
    feature_map_list = [conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2]

    #8X8X64
    return conv3_2,wts_list,feature_map_list


def decoder(inputs):
    scope = 'decoder'

    #8X8X64
    shape = tf.shape(inputs)
    output_shape = [shape[0],16,16,32]
    conv1_1, conv_w1_1, conv_b1_1 = ld.deconv_layer(inputs, 3, output_shape,"decode_1_1",stride=2,padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")
    
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
    x = tf.placeholder(tf.float32,[None,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS])

    bottleneck,wts_list_encoder,feature_map_list_encoder = encoder(x)
    output_image_sigmoid,wts_list_decoder,feature_map_list_decoder = decoder(bottleneck)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_image_sigmoid,labels=x)
    #loss  = tf.square(output_image_sigmoid-x)
    loss = tf.reduce_mean(loss)

    tf.summary.image('output_image',output_image_sigmoid)
    tf.summary.image('input_image',x)
    tf.summary.scalar('loss', loss)

    

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
        

        while start_count+BATCH_SIZE < 10000:
            selected_frames_batch = []
            for video_index in range(start_count,start_count+BATCH_SIZE):
                selected_frames = mnsit_data[video_index][:]/255.0
                selected_frames = np.array(selected_frames).reshape(-1,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
                selected_frames_batch.append(selected_frames)

            selected_frames_batch = np.array(selected_frames_batch).reshape(-1,IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNELS)
            _,generated_loss,summary = sess.run([train_op,loss,tf_tensorboard],feed_dict={x: selected_frames_batch})
        
            print("LOSS: " + str(generated_loss))

            summary_writer.add_summary(summary, summary_writer_it)
            summary_writer_it += 1
            start_count += BATCH_SIZE



if __name__ == "__main__":
    train()
