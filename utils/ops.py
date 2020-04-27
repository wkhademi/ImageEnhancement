def batch_normalization(input, training, decay=0.999, eps=1e-3):
    """
    Compute batch normalization on the input. If training use the batch mean and
    batch variance. If testing use the population mean and population variance.
    """
    offset = tf.Variable(tf.zeros(input.get_shape().as_list()[1:]), name='beta')
    scale = tf.Variable(tf.ones(), name='gamma')
    population_mean = tf.Variable(tf.zeros())
    population_var = tf.Variable(tf.ones())

    batch_mean, batch_var = tf.nn.moments(input, axis=[0])
    train_mean = tf.assign(population_mean, decay*population_mean + (1-decay)*batch_mean)
    train_var = tf.assign(population_var, decay*population_var + (1-decay)*batch_var)

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input, batch_mean, batch_var,
                                             offset, scale, eps, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(input, population_mean, population_var,
                                         offset, scale, eps, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics)


def instance_normalization(input, eps=1e-9):
    """
    Compute instance normalization on the input.
    """
    mean, var = tf.nn.moments(input, axis=[1,2], keep_dims=True)

    return (input - mean) / tf.sqrt(var + eps)


def __weights_init(size,
                   in_channels,
                   out_channels,
                   init_type='normal',
                   init_gain=1.0):
    """
        Initialize weights given a specific initialization type.
        Args:
            size: Size of filter matrix
            in: # of channels for input
            out: # of channels desired for output
            init_type: Type of weight initialization
            init_gain: Scaling factor for weight initialization
        Returns:
            weights: Weight tensor
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
