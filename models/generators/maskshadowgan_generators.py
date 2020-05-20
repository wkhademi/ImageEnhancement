import tensorflow as tf
from utils import ops


class Generator:
    def __init__(self, channels=3, n_blocks=9, ngf=64, norm_type='instance',
                 init_type='normal', init_gain=1.0, training=True, name=None):
        self.channels = channels
        self.n_blocks = n_blocks
        self.ngf = ngf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.is_training = training
        self.name = name
        self.reuse = False

    def __call__(self, input, mask=None):
        """
        Resnet-based generator that contains Resnet blocks in between
        some downsampling and upsampling layers.
        """
        def resnet_block(input, channels, filter_size=3, stride=1, norm_type='instance',
                         is_training=True, scope=None, reuse=False):
            """
            Residual block that contains two 3x3 convolution layers with the same number
            of filters on both layer.
            """
            with tf.variable_scope(scope, reuse=reuse):
                conv1 = ops.conv(input, channels, channels, filter_size=filter_size, stride=stride,
                                 padding_type='REFLECT', norm_type=norm_type,
                                 is_training=is_training, scope='res_conv1', reuse=reuse)

                conv2 = ops.conv(conv1, channels, channels, filter_size=filter_size, stride=stride,
                                 padding_type='REFLECT', norm_type=norm_type, activation_type=None,
                                 is_training=is_training, scope='res_conv2', reuse=reuse)

                layer = input + conv2

                return layer

        with tf.variable_scope(self.name):
            channels = self.channels+1 if mask is not None else self.channels
            concat_input = tf.concat([input, mask], axis=-1) if mask is not None else input

            # 7x7 convolution-instance norm-relu layer with 64 filters and stride 1
            c7s1_64 = ops.conv(concat_input, in_channels=channels, out_channels=self.ngf, filter_size=7, stride=1,
                               padding_type='REFLECT', weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                               norm_type=self.norm_type, is_training=self.is_training, scope='c7s1-64', reuse=self.reuse)

            # 3x3 convolution-instance norm-relu layer with 128 filters and stride 2
            d128 = ops.conv(c7s1_64, in_channels=self.ngf, out_channels=2*self.ngf, filter_size=3,
                            stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                            norm_type=self.norm_type, is_training=self.is_training, scope='d128', reuse=self.reuse)

            # 3x3 convolution-instance norm-relu layer with 256 filters and stride 2
            d256 = ops.conv(d128, in_channels=2*self.ngf, out_channels=4*self.ngf, filter_size=3,
                            stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                            norm_type=self.norm_type, is_training=self.is_training, scope='d256', reuse=self.reuse)

            r256 = d256

            # Resnet blocks with 256 filters
            for idx in range(self.n_blocks):
                # residual block that contains two 3x3 convolution layers with 256 filters and stride 1
                r256 = resnet_block(r256, channels=4*self.ngf, filter_size=3, stride=1, norm_type=self.norm_type,
                                    is_training=self.is_training, scope='r256-'+str(idx), reuse=self.reuse)

            # 3x3 fractional strided convolution-instance norm-relu layer with 128 filters and stride 1/2
            u128 = ops.transpose_conv(r256, in_channels=4*self.ngf, out_channels=2*self.ngf, filter_size=3,
                                      stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                                      norm_type=self.norm_type, is_training=self.is_training, scope='u128', reuse=self.reuse)

            # 3x3 fractional strided convolution-instance norm-relu layer with 64 filters and stride 1/2
            u64 = ops.transpose_conv(u128, in_channels=2*self.ngf, out_channels=self.ngf, filter_size=3,
                                     stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                                     norm_type=self.norm_type, is_training=self.is_training, scope='u64', reuse=self.reuse)

            # 7x7 convolution-instance norm-relu layer with 3 filters and stride 1
            c7s1_3 = ops.conv(u64, in_channels=self.ngf, out_channels=self.channels, filter_size=7, stride=1,
                              padding_type='REFLECT', weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              use_bias=False, norm_type=None, activation_type=None, is_training=self.is_training, 
                              scope='c7s1-3', reuse=self.reuse)

            output = tf.math.tanh(c7s1_3+input, name='gen_out')

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
