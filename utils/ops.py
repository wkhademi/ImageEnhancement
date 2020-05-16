import tensorflow as tf


def conv(input, in_channels, out_channels, filter_size, stride, padding_type='SAME',
         weight_init_type='normal', weight_init_gain=1.0, use_bias=False,
         bias_const=0.0, norm_type='instance', activation_type='ReLU', slope=0.2,
         is_training=True, scope=None, reuse=False):
    """
    Convolution-Normalization-Activation layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        # weight initialization
        weights = __weights_init(filter_size, in_channels, out_channels,
                                 init_type=weight_init_type, init_gain=weight_init_gain)

        if padding_type == 'REFLECT': # add reflection padding to input
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
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # convolution
        layer = conv(upsampled_inputs, in_channels, out_channels, filter_size, stride,
                     padding_type=padding_type, weight_init_type=weight_init_type,
                     weight_init_gain=weight_init_gain, use_bias=use_bias, bias_const=bias_const,
                     norm_type=norm_type, activation_type=activation_type, slope=slope,
                     is_training=is_training, scope='upsample_conv', reuse=reuse)

        return layer


def transpose_conv(input, in_channels, out_channels, filter_size=3, stride=2,
                   weight_init_type='normal', weight_init_gain=1.0, use_bias=False,
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
        activation= tf.nn.leaky_relu(input, alpha=slope, name='leakyrelu')
    elif activation_type == 'tanh':
        activation = tf.nn.tanh(input, name='tanh')
    elif activation_type == 'sigmoid':
        activation = tf.nn.sigmoid(input, name='sigmoid')
    else:
        activation = input

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

    batch_mean, batch_var = tf.nn.moments(input, axes=[0,1,2])
    train_mean = tf.assign(population_mean, decay*population_mean + (1-decay)*batch_mean)
    train_var = tf.assign(population_var, decay*population_var + (1-decay)*batch_var)

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input, batch_mean, batch_var,
                                             beta, gamma, eps, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(input, population_mean, population_var,
                                         beta, gamma, eps, name='batch_norm')

    return tf.cond(is_training, batch_statistics, population_statistics)


def __instance_normalization(input, init_gain=0.02, eps=1e-9):
    """
    Compute instance normalization on the input.
    """
    with tf.variable_scope('instance_norm'):
        channels = input.get_shape().as_list()[3]
        scale = tf.get_variable('weights', shape=[channels], dtype=tf.float32,
                                initializer=tf.initializers.truncated_normal(stddev=init_gain))
        offset = __biases_init(channels)
        mean, var = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        norm = scale * ((input - mean) / tf.sqrt(var + eps)) + offset

    return norm 


def __fixed_padding(filter_size):
    """
    Calculate padding needed to keep input from being downsampled.
    """
    pad_total = filter_size - 1
    pad = pad_total // 2
    padding = [[0,0], [pad, pad], [pad, pad], [0, 0]]

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

    weights = tf.get_variable("weights", shape=[size, size, in_channels, out_channels],
                              dtype=tf.float32, initializer=init)

    return weights


def __biases_init(size, constant=0.0):
    """
    Initialize biases to a given constant.
    """
    biases = tf.get_variable("biases", shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(constant))

    return biases
