import tensorflow as tf


def conv(input, in_channels, out_channels, filter_size, stride, padding_type='SAME',
         weight_init_type='normal', weight_init_gain=1.0, use_bias=True,
         bias_const=0.0, norm_type='instance', activation_type='ReLU', slope=0.2,
         is_training=True, scope=None, reuse=False):
    """
    Convolution-Normalization-Activation layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        # weight initialization
        weights = __weights_init(filter_size, in_channels, out_channels,
                                 init_type=weight_init_type, init_gain=weight_init_gain)

        if padding_type == 'REFLECT':  # add reflection padding to input
            padding = __fixed_padding(filter_size)
            padded_input = tf.pad(input, padding, padding_type)
            padding_type = 'VALID'
        elif padding_type == 'VALID':
            padded_input = input
        elif padding_type == 'SAME':
            padded_input = input

        layer = tf.nn.conv2d(padded_input, weights, strides=[1, stride, stride, 1],
                             padding=padding_type)

        if use_bias:
            biases = __biases_init(out_channels, constant=bias_const)
            layer = tf.nn.bias_add(layer, biases)

        # instance, batch, or no normalization
        layer = __normalization(layer, init_gain=weight_init_gain,
                                is_training=is_training, norm_type=norm_type)

        # relu, leaky relu, or no activation
        layer = __activation_fn(layer, slope=slope, activation_type=activation_type)

    return layer


def upsample(input, rescale_factor, in_channels, out_channels, filter_size, stride,
             padding_type='SAME', weight_init_type='normal', weight_init_gain=1.0,
             use_bias=False, bias_const=0.0, norm_type='instance', activation_type='ReLU',
             slope=0.2, is_training=True, scope=None, reuse=False):
    """
    Upsample-Convolution layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out_shape = rescale_factor * input.get_shape().as_list()[1]

        # upsample images by rescale_factor
        upsampled_inputs = tf.image.resize_images(input, [out_shape, out_shape],
                                                  method=tf.image.ResizeMethod.BILINEAR)

        # convolution
        layer = conv(upsampled_inputs, in_channels, out_channels, filter_size, stride,
                     padding_type=padding_type, weight_init_type=weight_init_type,
                     weight_init_gain=weight_init_gain, use_bias=use_bias, bias_const=bias_const,
                     norm_type=norm_type, activation_type=activation_type, slope=slope,
                     is_training=is_training, scope='upsample_conv', reuse=reuse)

        return layer


def transpose_conv(input, in_channels, out_channels, filter_size=3, stride=2,
                   weight_init_type='normal', weight_init_gain=1.0, use_bias=True,
                   bias_const=0.0, norm_type='instance', activation_type='ReLU',
                   is_training=True, scope=None, reuse=False):
    """
    TransposeConvolution-Normalization-Activation layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = input.get_shape().as_list()
        out_shape = 2*shape[1]
        batch_size = tf.shape(input)[0]

        # weight initialization
        weights = __weights_init(filter_size, out_channels, in_channels,
                                 init_type=weight_init_type, init_gain=weight_init_gain)

        layer = tf.nn.conv2d_transpose(input, weights,
                                       output_shape=[batch_size, out_shape, out_shape, out_channels],
                                       strides=[1, stride, stride, 1], padding='SAME')

        if use_bias:
            biases = __biases_init(out_channels, constant=bias_const)
            layer = tf.nn.bias_add(layer, biases)

        # instance, batch, or no normalization
        layer = __normalization(layer, init_gain=weight_init_gain,
                                is_training=is_training, norm_type=norm_type)

        # relu, leaky relu, or no activation
        layer = __activation_fn(layer, activation_type=activation_type)

    return layer


def dense(input, in_size, out_size, weight_init_type='normal', weight_init_gain=1.0,
          use_bias=True, bias_const=0.0, norm_type='instance', activation_type='ReLU',
          is_training=True, scope=None, reuse=False):
    """
    FullyConnected-Normalization-Activation layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        weights = __weights_init(None, in_size, out_size, init_type=weight_init_type,
                                 init_gain=weight_init_gain)

        layer = tf.matmul(input, weights)

        if use_bias:
            biases = __biases_init(out_size, constant=bias_const)
            layer = tf.nn.bias_add(layer, biases)

        # instance, batch, or no normalization
        layer = __normalization(layer, init_gain=weight_init_gain,
                                is_training=is_training, norm_type=norm_type)

        # relu, leaky relu, or no activation
        layer = __activation_fn(layer, activation_type=activation_type)

    return layer


def pixel_shuffle(input, block_size=2):
    """
    Upsample inputs by "block_size" using pixel shuffle
    """
    upsampled = tf.nn.depth_to_space(input, block_size, name='pixelshuffle_upsample')

    return upsampled


def flatten(input):
    """
    Flatten a tensor to [batch_size, -1]
    """
    shape = input.get_shape().as_list()

    flattened = tf.reshape(input, [-1, shape[1]*shape[2]*shape[3]])

    return flattened


def max_pooling(input, filter_size=2, stride=2, padding='SAME'):
    """
    Perform max pooling on input.
    """
    pool = tf.nn.max_pool2d(input, filter_size, stride, padding)

    return pool


def average_pooling(input, filter_size=2, stride=2):
    """
    Perform average pooling on input.
    """
    pool = tf.nn.avg_pool2d(input, filter_size, stride)

    return pool


def global_average_pooling(input):
    """
    Compute mean across the height and width dimensions for each channel of
    every image.
    """
    pool = tf.reduce_mean(input, axis=[1, 2])

    return pool


def pad_tensor(input, divide=16):
    """
    Pad input tensor.
    """
    shape = input.get_shape().as_list()
    height = shape[1]
    width = shape[2]

    if width % divide != 0 or height % divide != 0:
        width_res = width % divide
        height_res = height % divide

        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        input = tf.pad(input, tf.constant(0, [[pad_top, pad_bottom], [pad_left, pad_right]]))
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    shape = input.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    """
    Remove padding from tensor.
    """
    shape = input.get_shape().as_list()
    height = shape[1]
    width = shape[2]

    input = tf.image.crop_to_bounding_box(input, pad_top, pad_left, height-(pad_top+pad_bottom), width-(pad_left+pad_right))

    return input


def crop(input, height, width, patch_size):
    """
    Crop a patch out of an image.
    """
    cropped_input = tf.image.crop_to_bounding_box(input, height, width, patch_size, patch_size)

    return cropped_input


def __normalization(input, init_gain=1.0, is_training=True, norm_type='instance'):
    """
    Normalization to be applied to layer.
    """
    if norm_type == 'batch':
        norm = __batch_normalization(input, is_training=is_training)
    elif norm_type == 'instance':
        norm = __instance_normalization(input, init_gain=init_gain)
    else:
        norm = input

    return norm


def __activation_fn(input, slope=0.2, activation_type='ReLU'):
    """
    Non-linear activation to be applied to layer.
    """
    if activation_type == 'ReLU':
        activation = tf.nn.relu(input, name='relu')
    elif activation_type == 'LeakyReLU':
        activation = tf.nn.leaky_relu(input, alpha=slope, name='leakyrelu')
    elif activation_type == 'ParametricReLU':
        activation = __parametric_relu(input, name='parametricrelu')
    elif activation_type == 'tanh':
        activation = tf.nn.tanh(input, name='tanh')
    elif activation_type == 'sigmoid':
        activation = tf.nn.sigmoid(input, name='sigmoid')
    else:
        activation = input

    return activation


def __parametric_relu(inputs, name='parametricrelu'):
    """
    Parametric ReLU activation function.
    """
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', shape=inputs.get_shape()[-1],
                                initializer=tf.constant_initializer(0.1), dtype=tf.float32)

        activation = tf.maximum(0., inputs) + alpha*tf.minimum(0., inputs)

    return activation


def __batch_normalization(input, is_training, decay=0.999, eps=1e-3):
    """
    Compute batch normalization on the input. If training use the batch mean and
    batch variance. If testing use the population mean and population variance.
    """
    shape = input.get_shape().as_list()[-1]  # get out channels
    beta = tf.Variable(tf.zeros(shape), name='beta')
    gamma = tf.Variable(tf.ones(shape), name='gamma')
    population_mean = tf.Variable(tf.zeros(shape))
    population_var = tf.Variable(tf.ones(shape))

    batch_mean, batch_var = tf.nn.moments(input, axes=[0, 1, 2])
    train_mean = tf.assign(population_mean, decay*population_mean + (1-decay)*batch_mean)
    train_var = tf.assign(population_var, decay*population_var + (1-decay)*batch_var)

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input, batch_mean, batch_var,
                                             beta, gamma, eps, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(input, population_mean, population_var,
                                         beta, gamma, eps, name='batch_norm')

    return tf.cond(tf.cast(is_training, tf.bool), batch_statistics, population_statistics)


def __instance_normalization(input, init_gain=0.02, eps=1e-9, name='weights'):
    """
    Compute instance normalization on the input.
    """
    with tf.variable_scope('instance_norm'):
        channels = input.get_shape().as_list()[3]
        scale = tf.get_variable(name, shape=[channels], dtype=tf.float32,
                                initializer=tf.initializers.truncated_normal(mean=1.0, stddev=init_gain))
        offset = __biases_init(channels, name=name+'_biases')
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        norm = scale * ((input - mean) / tf.sqrt(var + eps)) + offset

    return norm


def __fixed_padding(filter_size):
    """
    Calculate padding needed to keep input from being downsampled.
    """
    pad_total = filter_size - 1
    pad = pad_total // 2
    padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]

    return padding


def __weights_init(size, in_channels, out_channels, init_type='normal', init_gain=1.0):
    """
    Initialize weights given a specific initialization type.
    """
    if init_type == 'normal':
        init = tf.initializers.truncated_normal(stddev=init_gain)
    elif init_type == 'he':
        init = tf.initializers.he_normal()
    elif init_type == 'orthogonal':
        init = tf.initializers.orthogonal(gain=init_gain)

    if size is not None:  # convolution filter weights
        weights = tf.get_variable("weights", shape=[size, size, in_channels, out_channels],
                                  dtype=tf.float32, initializer=init)
    else:  # fully connected weights
        weights = tf.get_variable("weights", shape=[in_channels, out_channels],
                                  dtype=tf.float32, initializer=init)

    return weights


def __biases_init(size, constant=0.0, name='biases'):
    """
    Initialize biases to a given constant.
    """
    biases = tf.get_variable(name, shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(constant))

    return biases
