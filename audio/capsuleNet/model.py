#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from capsule_layers import conv_caps_layer, fully_connected_caps_layer
import tensorflow as tf
import json,glob,random
from layers import fc,create_conv,variable_summaries
import utils
import os
import time
from PIL import Image
'''
    FLOW:
        CapsuleNet.__init__
        train_model
        init
            build_model_inputs
            build_main_network
            build_loss
            init_session
        optimize
'''

'''
    LEFT:
        predict
        add train_log name and test_log name in hyperparameters.json
        tensorboard
'''




class CapsuleNet:

    # Numbers of label to predict
    #NB_OUTPUT_LABELS = 2



    def __init__(self,hyperparameters_dir):
        #Its better to give batch_size rather than None
        #image attributes

        #all hyperparameters are loaded from a json
        #this init function is only for static hyperparameters loaded from json

        #load json into a map
        print(hyperparameters_dir)
        hyperparameters = json.loads(open(hyperparameters_dir).read())
        print(hyperparameters)

        self.image_rows = hyperparameters['image_rows']
        self.image_columns = hyperparameters['image_columns']
        self.image_channels = hyperparameters['image_channels']

        self.conv1_size = hyperparameters['conv1_size']
        self.conv1_filters = hyperparameters['conv1_filters']

        self.conv2_size = hyperparameters['conv2_size']
        self.conv2_filters = hyperparameters['conv2_filters']

        self.conv_2_dropout = hyperparameters['drop_out_prob']

        self.caps1_size = hyperparameters['caps1_size']
        self.caps1_vec_len = hyperparameters['caps1_vec_len']
        self.caps1_nb_capsules = hyperparameters['caps1_nb_capsules'] 

        self.caps2_vec_len = hyperparameters['caps2_vec_len']
        self.caps2_nb_capsules = hyperparameters['caps2_nb_capsules']

        self.routing_steps = hyperparameters['routing_steps']

        self.learning_rate = hyperparameters['learning_rate']

        self.batch_size = hyperparameters['batch_size']
        #self.train_log_name = hyperparameters['train_log_name']
        #self.test_log_name = hyperparameters['test_log_name']

        self.number_of_epochs = 10000000
        self.train_log_name = './logs/train'
        self.test_log_name = './logs/test'
        self.NB_OUTPUT_LABELS=2


    def init(self):
        # Get graph inputs
        print("Building the model input")
        self.build_model_input()

        # Create the first convolution and the CapsNet
        print("building the capsules")
        self.tf_caps1, self.tf_caps2 = self.build_main_network()

        one_hot_labels = tf.one_hot(self.tf_labels, depth=self.NB_OUTPUT_LABELS)

        # Build the loss
        print("building the loss")
        loss = self.build_loss(
            self.tf_caps2, self.tf_labels,one_hot_labels)

        (self.tf_margin_loss_sum, self.tf_predicted_class,
         self.tf_correct_prediction, self.tf_accuracy, self.tf_margin_loss) = loss

        # Build optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.tf_optimizer = optimizer.minimize(self.tf_margin_loss, global_step=tf.Variable(0, trainable=False))
        #training_operation = slim.learning.create_train_op(self.tf_margin_loss, optimizer,summarize_gradients=True)

        # Log value into tensorboard
        grad_conv_w_1, grad_conv_b_1 = tf.gradients(ys=self.tf_margin_loss, 
            xs=[self.conv_w_1, self.conv_b_1])

        grad_caps_1 = tf.gradients(ys=self.tf_margin_loss, xs=[tf.get_collection(tf.GraphKeys.VARIABLES, 'CAPSULE1')[0]])
        
        grad_caps_w_2 = tf.gradients(ys=self.tf_margin_loss, 
            xs=[self.caps2_w])

        tf.summary.scalar('margin_loss', self.tf_margin_loss)
        tf.summary.scalar('accuracy', self.tf_accuracy)
        variable_summaries(grad_conv_w_1, 'grad_conv_w_1')
        variable_summaries(grad_conv_b_1, 'grad_conv_b_1')
        #variable_summaries(grad_conv_w_2, 'grad_conv_w_2')
        #variable_summaries(grad_conv_b_2, 'grad_conv_b_2')
        variable_summaries(grad_caps_1, 'grad_caps_1')
        variable_summaries(tf.get_collection(tf.GraphKeys.VARIABLES, 'CAPSULE1')[0], 'caps_1')
        variable_summaries(self.caps2_w, 'caps_2_w')
        variable_summaries(grad_caps_w_2, 'grad_caps_w_2')
        #variable_summaries(self.b_ij, 'b_ij')
        #variable_summaries(self.c_ij, 'c_ij')
        
        #self.tf_test = tf.random_uniform([2], minval=0, maxval=None, dtype=tf.float32, seed=None, name="tf_test")
        #bring from model_base
        self.init_session()



    def build_model_input(self):
        # Images image_rows*image_columns*image_channels
        self.tf_images = tf.placeholder(tf.float32, [self.batch_size, self.image_rows, self.image_columns, self.image_channels], name='images')
        self.tf_labels = tf.placeholder(tf.int64, [self.batch_size], name='labels')
        tf.summary.image("input_img", self.tf_images, max_outputs=self.batch_size)

    def build_main_network(self):
        #Layer1 : conv1
        shape1 = (self.conv1_size, self.conv1_size, self.image_channels, self.conv1_filters)
        conv1, self.conv_w_1, self.conv_b_1 = create_conv('1', self.tf_images, shape1, relu=True, max_pooling=False, padding='VALID')
        
        # Layer 2 : conv2
        #shape2 = (self.conv2_size, self.conv2_size, self.conv1_filters, self.conv2_filters)
        #conv2, self.conv_w_2, self.conv_b_2 = create_conv('2', conv1, shape2, relu=True, max_pooling=False, padding='VALID')

        #Layer 3 : dropout
        #conv2 = tf.nn.dropout(conv2, keep_prob=self.conv_2_dropout)

        # Layer 4 : first capsules layer
        caps1 = conv_caps_layer(
            input_layer = conv1,
            capsules_size = self.caps1_vec_len, #8
            nb_filters = self.caps1_nb_capsules, #32
            kernel = self.caps1_size) #9
        
        # Layer 5 : second capsules layer used to predict the output
        caps2,self.caps2_w = fully_connected_caps_layer(
            input_layer=caps1,
            capsules_size=self.caps2_vec_len, #16
            nb_capsules=self.caps2_nb_capsules, #10
            iterations=self.routing_steps)

        return caps1, caps2


    def build_loss(self, caps2, labels, one_hot_vector):

        # Get the length of each capsule. axis = 2 tells that this is along columns(16).
        absolute_capsules_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True))
        self.absolute_capsules_length = absolute_capsules_length
        max_l = tf.square(tf.maximum(0., 0.9 - absolute_capsules_length))
        print(max_l.shape)
        max_l = tf.reshape(max_l, shape=(-1, self.NB_OUTPUT_LABELS))
        max_r = tf.square(tf.maximum(0., absolute_capsules_length - 0.1))
        max_r = tf.reshape(max_r, shape=(-1, self.NB_OUTPUT_LABELS))
        

        t_c = one_hot_vector
        
        print(t_c.shape)
        print(max_l.shape)        
        print(max_r.shape)

        m_loss = t_c * max_l + 0.5 * (1 - t_c) * max_r
        margin_loss_sum = tf.reduce_sum(m_loss, axis=1)
        margin_loss = tf.reduce_mean(margin_loss_sum)

        # Accuracy
        predicted_class = tf.argmax(absolute_capsules_length, axis=1)
        predicted_class = tf.reshape(predicted_class, [tf.shape(absolute_capsules_length)[0]])
        correct_prediction = tf.equal(predicted_class, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return (margin_loss_sum, predicted_class, correct_prediction, accuracy,margin_loss)


    def optimize(self, images, labels, tb_save=True):
        tensors = [self.tf_optimizer, self.tf_margin_loss, self.tf_accuracy, self.tf_tensorboard, self.absolute_capsules_length,self.caps2_w]
        _, loss, acc, summary, absolute_capslen, cap2_weights = self.sess.run(tensors,
            feed_dict={
            self.tf_images: images,
            self.tf_labels: labels
        })

        print "absolute_capslen: " + str(absolute_capslen)

        if tb_save:
            # Write data to tensorboard
            self.train_writer.add_summary(summary, self.train_writer_it)
            self.train_writer_it += 1

        return loss, acc


    def init_session(self):
       
        print "creating sesssion"
        self.sess = tf.Session()
        # Tensorboard
        self.tf_tensorboard = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.train_log_name, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.test_log_name)
        #  Create session
        print "creating saver"
        self.saver = tf.train.Saver()
        
        print "Initializing global variables"
        # Init variables
        self.sess.run(tf.global_variables_initializer())
        print "Initialized global variables"

        # if self.load_session():
        #     print "Load Success"
        # else:
        #     print "Load Failed."
        

        self.train_writer_it = 0
        self.test_writer_it = 0


    def save_session(self, step):
        checkpoint_dir = "./checkpoint"
        model_name = "capsuleNet.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)



    def load_session(self):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = "./checkpoint"
        model_name = "capsuleNet.model"
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def print_validation_loss(self, images,labels, epoch, iteration):
        
        tensors = [self.tf_margin_loss, self.tf_accuracy]
        validation_loss, validation_accuracy = self.sess.run(tensors,
            feed_dict={
            self.tf_images: images,
            self.tf_labels: labels
        })

        #validation_loss = self.tf_margin_loss.eval({ self.tf_images: images,
        #                                    self.tf_labels: labels })

        #validation_accuracy = self.accuracy.eval({ self.tf_images: images,
        #                                     self.tf_labels: labels })
        
        print("Epoch: [%4d/%4d] time: %.4f, loss: %.8f, accuracy: %.8f" \
            % (epoch, iteration,
                time.time() - self.start_time, validation_loss, validation_accuracy))



    def print_train_loss(self, images,labels, epoch, iteration):

        tensors = [self.tf_margin_loss, self.tf_accuracy]
        train_loss, train_accuracy = self.sess.run(tensors,
            feed_dict={
            self.tf_images: images,
            self.tf_labels: labels
        })

        #train_loss = self.tf_margin_loss.eval({ self.tf_images: images,
        #                                    self.tf_labels: labels })

        #train_accuracy = self.accuracy.eval({ self.tf_images: images,
        #                                     self.tf_labels: labels })
        
        print("Epoch: [%4d/%4d] time: %.4f, loss: %.8f, accuracy: %.8f" \
            % (epoch, iteration,
                time.time() - self.start_time, train_loss, train_accuracy))
    

    def train_model(self):
        print("train_model")
        self.start_time=time.time()

        #data_train && data_validation are glob list of file_names
        #data_train,data_validation = utils.generate_and_split_spectograms_for_complete_data('../../dataset/audio')

        print "Loading validation data"
        #will load the images from file_names for complete validation_set
        #validation_images,validation_labels = utils.load_data(self.image_rows,self.image_columns,self.image_channels,data_validation)

        data_train = [('/Users/duggals/Downloads/one.jpeg',1),('/Users/duggals/Downloads/zero.jpeg',0)]

        iterations = (len(data_train))//self.batch_size
        print "length of trainind dataset: " + str(len(data_train))
        print "Number of iterations for eac epoch: " + str(iterations)
        self.init()

        epoch = 0
        counter = 0
        print "Running epochs now"
        while(epoch<self.number_of_epochs):
            epoch+=1
            print "Shuffling training data"
            random.shuffle(data_train)
            index=0
            for i in range(iterations):
                train_batch = data_train[index:index+self.batch_size]
                index+=self.batch_size
                print "Loading batch data"
                images,labels = utils.load_data(self.image_rows,self.image_columns,self.image_channels,train_batch)
                print "Training batch data size: " + str(len(images))
                
                Image.fromarray(images[0]).save("/Users/duggals/Downloads/one_train.jpeg")
                #print(images)
                #print(labels)
                self.optimize(images,labels)
                counter+=1
                if np.mod(counter,1)==0:
                    self.save_session(counter)
                    self.print_train_loss(images,labels, epoch, i)
                    #self.print_validation_loss(validation_images,validation_labels, epoch, i)


                    #tensorboard graphs
