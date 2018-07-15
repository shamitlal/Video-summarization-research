import os
import numpy as numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import layer_def as ld
import BasicConvLSTMCell

LEARNING_RATE = 0.005

def encoder(inputs):
    scope = 'encoder'
    inputs = tf.reshape(inputs,[-1,28,28,1])

    conv1_1, conv_w1_1, conv_b1_1 = ld.conv_layer(inputs, 3, 2, 16, "encode_1_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_1, "conv1_1")
    conv1_2 =conv1_1
    #conv1_2, conv_w1_2, conv_b1_2 = ld.conv_layer(conv1_1, 3, 2, 32, "encode_1_2",dilation=1,padding="SAME")
    ld.variable_summaries(conv1_2, "conv1_2")

    #14X14X32
    conv2_1, conv_w2_1, conv_b2_1 = ld.conv_layer(conv1_2, 3, 2, 32,"encode_2_1",dilation=1,padding="SAME")
    ld.variable_summaries(conv2_1, "conv2_1")
    conv2_2 = conv2_1
    #conv2_2, conv_w2_2, conv_b2_2 = ld.conv_layer(conv2_1, 3, 2, 64,"encode_2_2",padding="SAME")
    ld.variable_summaries(conv2_2, "conv2_2")

    #7X7X64
    #conv3_1, conv_w3_1, conv_b3_1 = ld.conv_layer(conv2_2, 3, 1, 32,"encode_3_1",dilation=1,padding="SAME",linear=False)
    #ld.variable_summaries(conv3_1, "conv3_1")
    #conv3_2 = conv3_1
    #conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 1, 32,"encode_3_2", padding="SAME")
    #ld.variable_summaries(conv3_2, "conv3_2")

    wts_list = [conv_w1_1, conv_b1_1, conv_w2_1, conv_b2_1]
    feature_map_list = [conv1_1,conv1_2,conv2_1,conv2_2]


    return conv2_2,wts_list,feature_map_list


def decoder(inputs):
    scope = 'decoder'

    #7X7X128
    shape = tf.shape(inputs)
    output_shape = [shape[0],14,14,64]
    conv1_1, conv_w1_1, conv_b1_1 = ld.deconv_layer(inputs, 3, output_shape,"decode_1_1",stride=2,padding="SAME")
    ld.variable_summaries(conv1_1, scope + "_conv1_1")
    
    #14X14X64
    shape = tf.shape(conv1_1)
    output_shape = [shape[0],28,28,32]
    conv2_1, conv_w2_1, conv_b2_1 = ld.deconv_layer(conv1_1, 3, output_shape,"decode_2_1",stride=2,padding="SAME")
    ld.variable_summaries(conv2_1, scope + "_conv2_1")

    #28X28X32
    shape = tf.shape(conv2_1)
    output_shape = [shape[0],28,28,16]    
    conv3_1, conv_w3_1, conv_b3_1 = ld.deconv_layer(conv2_1, 3, output_shape,"decode_3_1",stride=1,padding="SAME")
    ld.variable_summaries(conv3_1, scope + "_conv3_1")
    
    #28X28X16
    conv3_2, conv_w3_2, conv_b3_2 = ld.conv_layer(conv3_1, 3, 1, 1,"decode_3_2", padding="SAME",linear=True)
    
    output_image = conv3_2
    conv3_2  = tf.nn.sigmoid(conv3_2)
    ld.variable_summaries(conv3_2, scope + "_conv3_2")

    wts_list = [conv_w1_1, conv_b1_1, conv_w2_1, conv_b2_1,conv_w3_1, conv_b3_1, conv_w3_2, conv_b3_2]
    feature_map_list = [conv1_1,conv2_1,conv3_1,conv3_2]

    conv3_2_reshaped = tf.reshape(conv3_2,[-1,784])

    #28X28X1
    return output_image,conv3_2,conv3_2_reshaped,wts_list,feature_map_list


def train():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32,[None,784])

    bottleneck,wts_list_encoder,feature_map_list_encoder = encoder(x)
    output_image,output_image_sigmoid,output_image_sigmoid_reshaped,wts_list_decoder,feature_map_list_decoder = decoder(bottleneck)

    x_reshaped = tf.reshape(x,[-1,28,28,1])
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_image_sigmoid,labels=x_reshaped)
    loss  = tf.square(output_image_sigmoid-x_reshaped)
    loss = tf.reduce_mean(loss)

    tf.summary.image('output_image',output_image_sigmoid)
    tf.summary.image('input_image',x_reshaped)
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
    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        _,generated_loss,summary = sess.run([train_op,loss,tf_tensorboard],feed_dict={x: batch_xs})
        
        print("LOSS: " + str(generated_loss))

        summary_writer.add_summary(summary, summary_writer_it)
        summary_writer_it += 1



if __name__ == "__main__":
    train()
