#########This script defines the network architecure of the fusion network###########
import tensorflow as tf
import numpy as np
from FunFuseAn import init_param


def network(mri_image, pet_image):
    with tf.variable_scope('network'):
        ######feature extraction layers######
        
        ###low frequency MRI layer###
        with tf.variable_scope('mri_lf_layer1'):
            weights=tf.get_variable("w_mri_lf_1",[9,9,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_mri_lf_1",[16],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(mri_image, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = tf.maximum(conv1, 0.2 * conv1) #Leaky RELU
            
        ###low frequency PET layer###
        with tf.variable_scope('pet_lf_layer1'):
            weights=tf.get_variable("w_pet_lf_1",[7,7,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_pet_lf_1",[16],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(pet_image, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = tf.maximum(conv2, 0.2 * conv2) #Leaky RELU  
           
        ###high frequency MRI layer 1 ###
        with tf.variable_scope('mri_hf_layer1'):
            weights=tf.get_variable("w_mri_hf_1",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_mri_hf_1",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(mri_image, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = tf.maximum(conv3, 0.2 * conv3) #Leaky RELU
            
        ###high frequency PET layer 1 ###
        with tf.variable_scope('pet_hf_layer1'):
            weights=tf.get_variable("w_pet_hf_1",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_pet_hf_1",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(pet_image, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = tf.maximum(conv4, 0.2 * conv4) #Leaky RELU        
            
        ####high frequency MRI layer 2 ###
        with tf.variable_scope('mri_hf_layer2'):
            weights=tf.get_variable("w_mri_hf_2",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_mri_hf_2",[32],initializer=tf.constant_initializer(0.0))
            conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv5 = tf.maximum(conv5, 0.2 * conv5) #Leaky RELU
            
        ###high frequency PET layer 2 ###
        with tf.variable_scope('pet_hf_layer2'):
            weights=tf.get_variable("w_pet_hf_2",[5,5,16,32],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_pet_hf_2",[32],initializer=tf.constant_initializer(0.0))
            conv6= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv4, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv6 = tf.maximum(conv6, 0.2 * conv6) #Leaky RELU
            
        ###high frequency MRI layer 3 ###
        with tf.variable_scope('mri_hf_layer3'):
            weights=tf.get_variable("w_mri_hf_3",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_mri_hf_3",[64],initializer=tf.constant_initializer(0.0))
            conv7= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv5, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv7 = tf.maximum(conv7, 0.2 * conv7) #Leaky RELU
            
        ###high frequency PET layer 3 ###
        with tf.variable_scope('pet_hf_layer3'):
            weights=tf.get_variable("w_pet_hf_3",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_pet_hf_3",[64],initializer=tf.constant_initializer(0.0))
            conv8= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv6, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv8 = tf.maximum(conv8, 0.2 * conv8) #Leaky RELU
        
        ######1st Fusion rule: High frequency######
        fused_hf_features = tf.maximum(conv7, conv8) / (conv7 + conv8)
        
        ######reconstruction layer 1######
        with tf.variable_scope('recon_layer1'):
            weights=tf.get_variable("w_recon_1",[5,5,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_recon_1",[32],initializer=tf.constant_initializer(0.0))
            recon1= tf.contrib.layers.batch_norm(tf.nn.conv2d(fused_hf_features, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            recon1 = tf.maximum(recon1, 0.2 * recon1) #Leaky RELU
            
        ######reconstruction layer 2######
        with tf.variable_scope('recon_layer2'):
            weights=tf.get_variable("w_recon_2",[5,5,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias=tf.get_variable("b_recon_2",[16],initializer=tf.constant_initializer(0.0))
            recon2= tf.contrib.layers.batch_norm(tf.nn.conv2d(recon1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            recon2 = tf.maximum(recon2, 0.2 * recon2) #Leaky RELU
                        
        ######2nd Fusion rule: Low frequency features######
        fused_lf_features = (conv1 + conv2 + recon2)/3
            
        ######reconstruction layer 3######
        with tf.variable_scope('recon_layer3'):
            weights=tf.get_variable("w_recon_3",[5,5,16,1],initializer=tf.truncated_normal_initializer(stddev=1e-2))
            bias   =tf.get_variable("b_recon_3",[1],initializer=tf.constant_initializer(0.0))
            recon3 =tf.nn.conv2d(fused_lf_features, weights, strides=[1,1,1,1], padding='SAME') + bias
            recon3 =tf.nn.tanh(recon3)            
            
    return recon3
