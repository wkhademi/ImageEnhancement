import tensorflow as tf
from utils import ops


class Discriminator():
    def __init__(self, channels=3, ndf=64, norm_type='instance', init_type='normal',
                 init_gain=1.0, training=True, name=None):
        self.channels = channels
        self.ndf = ndf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.is_training = training
        self.name = name
        self.reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name):
            output = self.build_discriminator(input, self.channels, self.ndf, self.norm_type,
                                              self.init_type, self.init_gain, self.is_training)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def build_discriminator(self, input, channels=3, ndf=64, norm_type='batch',
                            init_type='normal', init_gain=0.02, is_training=True):
        """
        SRGAN Discriminator
        """
        conv_block1 = ops.conv(input, in_channels=channels, out_channels=ndf, filter_size=3,
                               stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=None, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv1', reuse=self.reuse)

        conv_block2 = ops.conv(conv_block1, in_channels=ndf, out_channels=ndf, filter_size=3,
                               stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv2', reuse=self.reuse)

        conv_block3 = ops.conv(conv_block2, in_channels=ndf, out_channels=2*ndf, filter_size=3,
                               stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv3', reuse=self.reuse)

        conv_block4 = ops.conv(conv_block3, in_channels=2*ndf, out_channels=2*ndf, filter_size=3,
                               stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv4', reuse=self.reuse)

        conv_block5 = ops.conv(conv_block4, in_channels=2*ndf, out_channels=4*ndf, filter_size=3,
                               stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv5', reuse=self.reuse)

        conv_block6 = ops.conv(conv_block5, in_channels=4*ndf, out_channels=4*ndf, filter_size=3,
                               stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv6', reuse=self.reuse)

        conv_block7 = ops.conv(conv_block6, in_channels=4*ndf, out_channels=8*ndf, filter_size=3,
                               stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv7', reuse=self.reuse)

        conv_block8 = ops.conv(conv_block7, in_channels=8*ndf, out_channels=8*ndf, filter_size=3,
                               stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv8', reuse=self.reuse)

        x = ops.flatten(conv_block8)

        dense1 = ops.dense()

        output = ops.dense()

        return output
