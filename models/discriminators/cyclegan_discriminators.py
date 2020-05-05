import tensorflow as tf
from utils import ops


class Discriminator():
    def __init__(self, channels=3, netD='basic', n_layers=3, ndf=64,
                 norm_type='instance', init_type='normal', init_gain=1.0,
                 training=True, gan_mode='lsgan', name=None):
        self.channels = channels
        self.netD = netD
        self.n_layers = n_layers
        self.ndf = ndf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.is_training = training
        self.gan_mode = gan_mode
        self.name = name
        self.reuse = False
        self.sigmoid = 'sigmoid' if gan_mode != 'lsgan' else None

    def __call__(self, input):
        with tf.variable_scope(self.name):
            if self.netD == 'basic': # 70x70 PatchGAN Discriminator
                output = self.n_layer_discriminator(input, self.channels, self.n_layers, self.ndf,
                                                    self.norm_type, self.init_type, self.init_gain,
                                                    self.is_training, self.sigmoid)
            elif self.netD == 'n_layers':
                output = self.n_layer_discriminator(input, self.channels, self.n_layers, self.ndf,
                                                    self.norm_type, self.init_type, self.init_gain,
                                                    self.is_training, self.sigmoid)
            elif self.netD == 'pixel': # 1x1 PatchGAN Discriminator
                output = self.pixel_discriminator(input, self.channels, self.ndf, self.norm_type,
                                                  self.init_type, self.init_gain, self.is_training, self.sigmoid)
            else:
                print("Invalid discriminator architecture.")
                sys.exit(1)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def n_layer_discriminator(self, input, channels=3, n_layers=3, ndf=64,
                              norm_type='instance', init_type='normal',
                              init_gain=1.0, is_training=True, sigmoid=None):
        """
        N-layer PatchGAN Discriminator
        """
        # first layer does not use instance normalization
        layer = ops.conv(input, in_channels=channels, out_channels=ndf, filter_size=4,
                         stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                         use_bias=False, norm_type=None, activation_type='LeakyReLU',
                         is_training=is_training, scope='layer0', reuse=self.reuse)

        nf_mult = 1
        nf_mult_prev = 1
        for idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** idx, 8)

            # perform 4x4 convolutions for n layers with a max of 512 filters
            layer = ops.conv(layer, in_channels=ndf*nf_mult_prev, out_channels=ndf*nf_mult,
                             filter_size=4, stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                             norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                             scope='layer'+str(idx), reuse=self.reuse)

        # nth layer of 4x4 convolutions uses a stride of 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layer = ops.conv(layer, in_channels=ndf*nf_mult_prev, out_channels=ndf*nf_mult,
                         filter_size=4, stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                         norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                         scope='layer'+str(n_layers), reuse=self.reuse)

        # produces a single channel prediction map
        layer = ops.conv(layer, in_channels=ndf*nf_mult, out_channels=1, filter_size=4, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, use_bias=False, norm_type=None,
                         activation_type=sigmoid, is_training=is_training, scope='d_out', reuse=self.reuse)

        return layer

    def pixel_discriminator(self, input, channels=3, ndf=64, norm_type='instance',
                            init_type='normal', init_gain=1.0, is_training=True,
                            sigmoid=None):
        """
        1x1 PatchGAN Discriminator (pixelGAN)
        """
        # 1x1 convolution with 64 filters and no normalization
        layer = ops.conv(input, in_channels=channels, out_channels=ndf, filter_size=1,
                         stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                         use_bias=False, norm_type=None, activation_type='LeakyReLU',
                         is_training=is_training, scope='layer0', reuse=self.reuse)

        # 1x1 convolution with 128 filters and instance normalization
        layer = ops.conv(layer, in_channels=ndf, out_channels=2*ndf, filter_size=1,
                         stride=1, weight_init_type=init_type, weight_init_gain=init_gain,
                         norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                         scope='layer1', reuse=self.reuse)

        # produces a single channel prediction map
        layer = ops.conv(layer, in_channels=2*ndf, out_channels=1, filter_size=1, stride=1,
                         weight_init_type=init_type, weight_init_gain=init_gain, norm_type=None,
                         activation_type=sigmoid, is_training=is_training, scope='layer2', reuse=self.reuse)

        return layer
