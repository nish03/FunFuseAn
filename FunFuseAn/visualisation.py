###########This script implements the visualisation framework of the FunFuseAn##########

#import relevant packages
import tensorflow as tf
import numpy as np
import cv2
from  skimage import color
from FunFuseAn.init_param import init_param
from FunFuseAn.network import network
from FunFuseAn.test_preprocess import test_preprocess_mri, test_preprocess_pet


#define placeholders
image_length, image_width, gray_channels, batch_size, epoch, lr, pet_image, mri_image = init_param()

#define the network
fused_image = network(mri_image,pet_image)

#import test images
test_mri = test_preprocess_mri(image_width,image_length)
test_pet = test_preprocess_pet(image_width,image_length)

#define gradients of the fused image
grad_mri = []
grad_pet = []
fused = []

#calculate gradients by placing it in a placeholder
saver = tf.train.Saver()
gradients_mri = tf.gradients(fused_image, images_mri)
gradients_pet = tf.gradients(fused_image, images_pet)

#run the test session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/home/..../Checkpoint_lambda_0/')
    for idx in range(0, len(test_mri[:,0,0,0])):
        batch_test_mri = test_mri[idx*batch_size : (idx+1)*batch_size,:,:,:]
        batch_test_pet = test_pet[idx*batch_size : (idx+1)*batch_size,:,:,:]
        grad_mr, grad_pe, fuse = sess.run([gradients_mri, gradients_pet, fused_image], feed_dict={images_mri: batch_test_mri, 
                                                                                                images_pet: batch_test_pet})
        grad_mri.append(grad_mr)
        grad_pet.append(grad_pe)
        fused.append(fuse)   
    fused = np.squeeze(np.asarray(fused))

#normalise and squeeze the fused image and the gradient maps
fused = np.squeeze((fused - np.min(fused)) / (np.max(fused) - np.min(fused)))
grad_mri = np.squeeze(np.asarray(grad_mri))
grad_mri = (grad_mri - np.min(grad_mri)) / (np.max(grad_mri) - np.min(grad_mri))
grad_pet = np.squeeze(np.asarray(grad_pet))
grad_pet = (grad_pet - np.min(grad_pet)) / (np.max(grad_pet) - np.min(grad_pet))

#Stack a single MRI gradient map into three channels
grad_mri_ = np.uint8(cv2.normalize(grad_mri[0,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
grad_mri_color = np.dstack((grad_mri_, grad_mri_, grad_mri_))

#Color code the PET gradient map
grad_pet_ = np.uint8(cv2.normalize(grad_pet[0,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
grad_pet_color = cv2.applyColorMap(grad_pet_, cv2.COLORMAP_JET)

#Convert RGB to HSV
grad_mri_hsv = color.rgb2hsv(grad_mri_color)
grad_pet_hsv = color.rgb2hsv(grad_pet_color)

#Assign Hue and Saturation channels of PET gradient map to Hue and Saturation of MRI gradient map 
omega = 0.6
grad_mri_hsv[..., 0] = grad_pet_hsv[..., 0] 
grad_mri_hsv[..., 1] = grad_pet_hsv[..., 1] * omega

#Convert HSV to RGB Fused image with color coded visualisation
fused_vis = color.hsv2rgb(grad_mri_hsv)
