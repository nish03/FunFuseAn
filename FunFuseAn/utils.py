########This script contains the utility functions for defining the SSIM function########

#import relevant packages
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def convert_image_dtype(
    image,
    dtype,
    saturate=False,
    name=None,
    ):

    image = ops.convert_to_tensor(image, name='image')
    if dtype == image.dtype:
        return array_ops.identity(image, name=name)

    with ops.name_scope(name, 'convert_image', [image]) as name:
        if image.dtype.is_integer and dtype.is_integer:
            scale_in = image.dtype.max
            scale_out = dtype.max
            if scale_in > scale_out:
                scale = (scale_in + 1) // (scale_out + 1)
                scaled = math_ops.div(image, scale)
                if saturate:
                    return math_ops.saturate_cast(scaled, dtype,
                            name=name)
                else:
                    return math_ops.cast(scaled, dtype, name=name)
            else:
                if saturate:
                    cast = math_ops.saturate_cast(image, dtype)
                else:
                    cast = math_ops.cast(image, dtype)
                scale = (scale_out + 1) // (scale_in + 1)
                return math_ops.multiply(cast, scale, name=name)
        elif image.dtype.is_floating and dtype.is_floating:
            return math_ops.cast(image, dtype, name=name)
        else:
            if image.dtype.is_integer:
                cast = math_ops.cast(image, dtype)
                scale = 1. / image.dtype.max
                return math_ops.multiply(cast, scale, name=name)
            else:
                scale = dtype.max + 0.5  # avoid rounding problems in the cast
                scaled = math_ops.multiply(image, scale)
                if saturate:
                    return math_ops.saturate_cast(scaled, dtype,
                            name=name)
                else:
                    return math_ops.cast(scaled, dtype, name=name)

def _verify_compatible_image_shapes(img1, img2):
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].assert_is_compatible_with(shape2[-3:])

    if shape1.ndims is not None and shape2.ndims is not None:
        for (dim1, dim2) in zip(reversed(shape1[:-3]),
                                reversed(shape2[:-3])):
            if not (dim1 == 1 or dim2 == 1
                    or dim1.is_compatible_with(dim2)):
                raise ValueError('Two images are not compatible: %s and %s'
                                  % (shape1, shape2))

    (shape1, shape2) = array_ops.shape_n([img1, img2])
    checks = []
    checks.append(control_flow_ops.Assert(math_ops.greater_equal(array_ops.size(shape1),
                  3), [shape1, shape2], summarize=10))
    checks.append(control_flow_ops.Assert(math_ops.reduce_all(math_ops.equal(shape1[-3:],
                  shape2[-3:])), [shape1, shape2], summarize=10))
    return (shape1, shape2, checks)

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03

def _ssim_helper(
    x,
    y,
    reducer,
    max_val,
    alpha,
    beta_gamma,
    compensation=1.0,
    ):

    c1 = (_SSIM_K1 * max_val) ** 2
    c2 = (_SSIM_K2 * max_val) ** 2
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)
    luminance = luminance ** alpha
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    cs = cs ** beta_gamma
    return (luminance, cs)


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""

    size = ops.convert_to_tensor(size, dtypes.int32)
    sigma = ops.convert_to_tensor(sigma)
    coords = math_ops.cast(math_ops.range(size), sigma.dtype)
    coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0
    g = math_ops.square(coords)
    g *= -0.5 / math_ops.square(sigma)
    g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g,
            shape=[-1, 1])
    g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = nn_ops.softmax(g)
    return array_ops.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1, img2, alpha, beta_gamma, max_val=1.0):
    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)

    (shape1, shape2) = array_ops.shape_n([img1, img2])
    checks =         [control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(shape1[-3:-1],
         filter_size)), [shape1, filter_size], summarize=8),
         control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(shape2[-3:-1],
         filter_size)), [shape2, filter_size], summarize=8)]
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    compensation = 1.0
    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1],
                              shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
                                padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-3],
                                 array_ops.shape(y)[1:]], 0))

    (luminance, cs) = _ssim_helper(img1, img2, reducer, max_val, alpha, beta_gamma,
                                   compensation)
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    cs = math_ops.reduce_mean(cs, axes)
    return (ssim_val, cs)
