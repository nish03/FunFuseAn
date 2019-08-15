###############This script trains the network ################

#import the relevant packages
import tensorflow as tf
from FunFuseAn.ssim_function import ssim
from FunFuseAn.init_param import init_param
from FunFuseAn.network import network
from FunFuseAn.train_preprocess import train_preprocess_mri, train_preprocess_pet

#define placeholders
image_length, image_width, gray_channels, batch_size, epoch, lr, images_pet, images_mri = init_param()

#import train dataset
train_mri = train_preprocess_mri(image_width, image_length)
train_pet = train_preprocess_pet(image_width, image_length)

#define the network 
fused_image = network(images_mri,images_pet)

#define lamda as fusion hyperparameter
lamda = 0

#define the loss functions
mri_ssim_loss = lamda * (1-  ssim(fused_image,images_mri, alpha = 1, beta_gamma = 1, max_val=1.0))
pet_ssim_loss = lamda * (1-  ssim(fused_image,images_pet,  alpha = 1,beta_gamma = 1, max_val=1.0))
mri_l2_loss =  (1-lamda)*(tf.reduce_mean(tf.square(fused_image - images_mri)))
pet_l2_loss =  (1-lamda)*(tf.reduce_mean(tf.square(fused_image - images_pet)))      
total_loss =  mri_ssim_loss +  pet_ssim_loss + mri_l2_loss + pet_l2_loss

#initialise the trainable variables
t_vars = tf.trainable_variables()
G_vars = [var for var in t_vars if 'network' in var.name] 
G_optim = tf.train.AdamOptimizer(lr).minimize(total_loss,  var_list= G_vars)

#initialise the training parameters
counter = 0
start_time = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#define lists to place intermediate training loss values
ep_mri_ssim_Loss = []
ep_pet_ssim_Loss = []
ep_mri_l2_Loss = []
ep_pet_l2_Loss = []

#start training
print("Training....")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for ep in range(epoch):
        mri_ssim_Loss = []
        pet_ssim_Loss = []
        mri_l2_Loss = []
        pet_l2_Loss = []
        #run batch images
        for idx in range(0, len(train_mri[:,0,0,0])):
            batch_images_mri = train_mri[idx*batch_size : (idx+1)*batch_size,:,:,:]
            batch_images_pet = train_pet[idx*batch_size : (idx+1)*batch_size,:,:,:]
            counter += 1
            #run the session
            img, _, loss1, loss2, loss3, loss4, loss5  = sess.run([fused_image,
                                                                   G_optim,
                                                                   mri_ssim_loss,
                                                                   pet_ssim_loss,
                                                                   mri_l2_loss,
                                                                   pet_l2_loss,
                                                                   total_loss], 
                                                                 feed_dict={images_mri: batch_images_mri, 
                                                                            images_pet: batch_images_pet}
                                                                 )
            mri_ssim_Loss.append(loss1)
            pet_ssim_Loss.append(loss2)
            mri_l2_Loss.append(loss3)
            pet_l2_Loss.append(loss4)
            if counter % 100 == 0:
                print("Epoch: [%2d],step: [%2d],time: [%4.4f], mri_ssim_loss: [%.8f], pet_ssim_loss: [%.8f], mri_l2_loss: [%.8f], pet_l2_loss: [%.8f]"                      % ((ep+1), counter, time.time()-start_time, loss1, loss2, loss3, loss4))  
        
        av_mri_ssim_Loss = np.average(mri_ssim_Loss)
        ep_mri_ssim_Loss.append(av_mri_ssim_Loss)
        
        av_pet_ssim_Loss = np.average(pet_ssim_Loss)
        ep_pet_ssim_Loss.append(av_pet_ssim_Loss)
        
        av_mri_l2_Loss = np.average(mri_l2_Loss)
        ep_mri_l2_Loss.append(av_mri_l2_Loss)
        
        av_pet_l2_Loss = np.average(pet_l2_Loss)
        ep_pet_l2_Loss.append(av_pet_l2_Loss)
        
        if(ep == epoch -1):
            saver.save(sess, '/home/..../Checkpoint_lambda_0/')
