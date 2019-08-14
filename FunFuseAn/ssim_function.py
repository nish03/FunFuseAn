##############This code defines the SSIM function used as the loss function in FunFuseAn###############

#import relevant tensorflow packages
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from FunFuseAn.utils import convert_image_dtype, _verify_compatible_image_shapes
from FunFuseAn.utils import _ssim_helper, _fspecial_gauss, _ssim_per_channel


#define the SSIM function
def ssim(img1, img2, alpha, beta_gamma, max_val):
    (_, _, checks) = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    img1 = convert_image_dtype(img1, dtypes.float32)
    img2 = convert_image_dtype(img2, dtypes.float32)
    (ssim_per_channel, _) = _ssim_per_channel(img1, img2, alpha, beta_gamma, max_val)
    #Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1])
