##########This script runs the test session of the fusion network###########

#import relevant packages
import tensorflow as tf
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

#run the test session
fused_test_image = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/home/....../Checkpoint_lambda_0/')
    for idx in range(0, len(test_mri[:,0,0,0])):
        batch_test_mri = test_mri[idx*batch_size : (idx+1)*batch_size,:,:,:]
        batch_test_pet = test_pet[idx*batch_size : (idx+1)*batch_size,:,:,:]
        img = sess.run(fused_image, feed_dict={images_mri: batch_test_mri, 
                                               images_pet: batch_test_pet})
        fused_test_image.append(img) 
    

