############define the hyperparameters#############

def init_param():
    #define the dimensions of the image
    image_length = 256
    image_width  = 256

    #define the number of channels for input data
    gray_channels = 1

    #define the batch size during training
    batch_size   = 1

    #define the number of epochs 
    epoch = 200

    #define the learning rate
    lr = 0.002
    
    #define input placeholders for the fusion network
    images_pet   =  tf.placeholder(tf.float32, [None, image_width,image_length,gray_channels],name='images_pet')
    images_mri   =  tf.placeholder(tf.float32, [None, image_width,image_length,gray_channels],name ='images_mri')
    
    return image_length, image_width, gray_channels, batch_size, epoch, lr, images_pet, images_mri
