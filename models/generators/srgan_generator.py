import tensorflow as tf
from utils import ops


class Generator:
    def __init__(self, channels=3, ngf=64, norm_type='instance', init_type='normal',
                 init_gain=1.0, dropout=False, training=True, name=None):
        self.channels = channels
        self.ngf = ngf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.dropout = dropout
        self.is_training = training
        self.name = name
        self.reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name):
            output = self.resnet_generator(input, self.channels, self.ngf, self.norm_type,
                                           self.init_type, self.init_gain, self.dropout,
                                           self.is_training)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def resnet_generator(self, input, channels=3, ngf=64, norm_type='instance',
                         init_type='normal', init_gain=1.0, dropout=False,
                         is_training=True, n_blocks=16):
        """
        Resnet-based generator that contains Resnet blocks in between
        some downsampling and upsampling layers.
        """
        def resnet_block(input, channels, filter_size=3, stride=1, norm_type='instance',
                         init_type='normal', init_gain=1.0, activation_type='ReLU',
                         is_training=True, dropout=False, scope=None, reuse=False):
            """
            Residual block that contains two 3x3 convolution layers with the same number
            of filters on both layer.
            """
            with tf.variable_scope(scope, reuse=reuse):
                conv1 = ops.conv(input, channels, channels, filter_size=filter_size, stride=stride,
                                 weight_init_type=init_type, weight_init_gain=init_gain, norm_type=norm_type,
                                 activation_type=activation_type, is_training=is_training, scope='res_conv1', reuse=reuse)

                if dropout:
                    conv1 = tf.nn.dropout(conv1, keep_prob=0.5)

                conv2 = ops.conv(conv1, channels, channels, filter_size=filter_size, stride=stride,
                                 weight_init_type=init_type, weight_init_gain=init_gain, norm_type=norm_type,
                                 activation_type=None, is_training=is_training, scope='res_conv2', reuse=reuse)

                layer = input + conv2

            return layer

        # 9x9 convolution-prelu layer with 64 filters and stride 1
        conv1 = ops.conv(input, in_channels=channels, out_channels=ngf, filter_size=9, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, norm_type=None,
                         activation_type='ParametricReLU', is_training=is_training, scope='conv1', reuse=self.reuse)

        res_block = conv1

        # n residual blocks
        for idx in range(n_blocks):
            # residual block that contains two 3x3 convolution layers with 64 filters and stride 1
            res_block = resnet_block(res_block, channels=ngf, filter_size=3, stride=1, norm_type=norm_type,
                                     init_type=init_type, init_gain=init_gain, activation_type='ParametricReLU',
                                     is_training=is_training, dropout=dropout, scope='res_block'+str(idx), reuse=self.reuse)

        # 3x3 convolution-prelu layer with 64 filters and stride 1
        conv2 = ops.conv(res_block, in_channels=ngf, out_channels=ngf, filter_size=3, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, norm_type=norm_type,
                         activation_type=None, is_training=is_training, scope='conv2', reuse=self.reuse)

        res = conv1 + conv2

        # 3x3 convolution layer with 256 filters and stride 1
        conv3 = ops.conv(res, in_channels=ngf, out_channels=4*ngf, filter_size=3, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, norm_type=None,
                         activation_type=None, is_training=is_training, scope='conv3', reuse=self.reuse)

        # pixel shuffle upsample by factor of 2
        upsample1 = ops.pixel_shuffle(conv3, block_size=2)
        upsample1 = ops.__parametric_relu(upsample1, name='parametricrelu1')

        # 3x3 convolution layer with 256 filters and stride 1
        conv4 = ops.conv(upsample1, in_channels=ngf, out_channels=4*ngf, filter_size=3, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, norm_type=None,
                         activation_type=None, is_training=is_training, scope='conv4', reuse=self.reuse)

        # pixel shuffle upsample by factor of 2
        upsample2 = ops.pixel_shuffle(conv4, block_size=2)
        upsample2 = ops.__parametric_relu(upsample2, name='parametricrelu2')

        # 9x9 convolution layer with 3 filters and stride 1
        output = ops.conv(upsample2, in_channels=4*ngf, out_channels=channels, filter_size=3, stride=1,
                          weight_init_type=init_type, weight_init_gain=init_gain, norm_type=None,
                          activation_type='tanh', is_training=is_training, scope='output', reuse=self.reuse)

        return output
