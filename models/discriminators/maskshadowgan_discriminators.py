import tensorflow as tf
from utils import ops


class Discriminator():
    def __init__(self, channels=3, ndf=64, norm_type='instance',
                 init_type='normal', init_gain=1.0, training=True, name=None):
        self.channels = channels
        self.ndf = ndf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.is_training = training
        self.name = name
        self.reuse = False

    def __call__(self, input):
        """
        70x70 PatchGAN discriminator w/ global average pooling at final layer
        """
        with tf.variable_scope(self.name):
            output = ops.conv(input, in_channels=self.channels, out_channels=self.ndf, filter_size=4,
                              stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              norm_type=None, activation_type='LeakyReLU', is_training=self.is_training,
                              scope='layer0', reuse=self.reuse)

            output = ops.conv(output, in_channels=self.ndf, out_channels=2*self.ndf, filter_size=4,
                              stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              norm_type=self.norm_type, activation_type='LeakyReLU', is_training=self.is_training,
                              scope='layer1', reuse=self.reuse)

            output = ops.conv(output, in_channels=2*self.ndf, out_channels=4*self.ndf, filter_size=4,
                              stride=2, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              norm_type=self.norm_type, activation_type='LeakyReLU', is_training=self.is_training,
                              scope='layer2', reuse=self.reuse)

            output = ops.conv(output, in_channels=4*self.ndf, out_channels=8*self.ndf, filter_size=4,
                              stride=1, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              norm_type=self.norm_type, activation_type='LeakyReLU', is_training=self.is_training,
                              scope='layer3', reuse=self.reuse)

            output = ops.conv(output, in_channels=8*self.ndf, out_channels=1, filter_size=4,
                              stride=1, weight_init_type=self.init_type, weight_init_gain=self.init_gain,
                              norm_type=None, activation_type=None, is_training=self.is_training,
                              scope='out', reuse=self.reuse)

            output = ops.global_average_pooling(output)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
