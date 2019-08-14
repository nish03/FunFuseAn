##############This script preprocesses the testing data###############

#import the relevant packages
import os
import natsort
import glob
import numpy as np
import imageio
from FunFuseAn.init_param import init_param

def test_preprocess_mri():
    filenames = os.listdir('/home/......./Testing/MRI/')
    dataset = os.path.join(os.getcwd(), '/home/......./Testing/MRI/')
    data = []
    for ext in ('*.gif', '*.png', '*.jpg','*.tif'):
        data.extend(glob.glob(os.path.join(dataset, ext)))
    data.sort(key=lambda x:int(filter(str.isdigit, x)))
    test_mri = np.zeros((len(data), image_width,image_length))
    for i in xrange(len(data)):
        test_mri[i,:,:] =(imageio.imread(data[i]))
        test_mri[i,:,:] =(test_mri[i,:,:] - np.min(test_mri[i,:,:])) / (np.max(test_mri[i,:,:]) - np.min(test_mri[i,:,:]))
        test_mri[i,:,:] = np.float32(test_mri[i,:,:])
    
    test_mri = test_mri[:,:,:,np.newaxis]
    return test_mri


def test_preprocess_pet():
    filenames = os.listdir('/home/......./Testing/PET/')
    dataset = os.path.join(os.getcwd(), '/home/......./Testing/PET/')
    data = []
    for ext in ('*.gif', '*.png', '*.jpg', '*.tif'):
        data.extend(glob.glob(os.path.join(dataset, ext)))
    data.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    test_oth = np.zeros((len(data),image_width,image_length))
    for i in xrange(len(data)):
        test_pet[i,:,:] =(imageio.imread(data[i]))
        test_pet[i,:,:] =(test_pet[i,:,:] - np.min(test_pet[i,:,:])) / (np.max(test_pet[i,:,:]) - np.min(test_pet[i,:,:]))
        test_pet[i,:,:] = np.float32(test_oth[i,:,:])
    
    test_pet = test_pet[:,:,:,np.newaxis]
    return test_pet
