##########This script runs the test session of the fusion network###########

#import relevant packages
import tensorflow as tf
from FunFuseAn.init_param import init_param
from FunFuseAn.network import network


image_length, image_width, gray_channels, batch_size, epoch, lr, pet_image, mri_image = init_param()

#run the test session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/home/....../Checkpoint_lambda_0/')
    img = sess.run(fused_image, feed_dict={images_mri: test_mri, 
                                           images_pet: test_pet}
                   )
