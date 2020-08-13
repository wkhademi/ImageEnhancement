import tensorflow as tf
from utils import ops


class Generator:
    def __init__(self, channels=3, netG='resnet_9blocks', ngf=64, norm_type='instance',
                 init_type='normal', init_gain=1.0, dropout=False, training=True,
                 self_attention=True, times_residual=True, skip=0.0, name=None):
        self.channels = channels
        self.netG = netG
        self.ngf = ngf
        self.norm_type = norm_type
        self.init_type = init_type
        self.init_gain = init_gain
        self.dropout = dropout
        self.is_training = training
        self.self_attention = self_attention
        self.times_residual = times_residual
        self.skip = skip
        self.name = name
        self.reuse = False

    def __call__(self, input, gray=None):
        with tf.variable_scope(self.name):
            if self.netG == 'resnet_9blocks':
                output = self.resnet_generator(input, self.channels, self.ngf, self.norm_type,
                                               self.init_type, self.init_gain, self.dropout,
                                               self.is_training, n_blocks=9)
            elif self.netG == 'resnet_6blocks':
                output = self.resnet_generator(input, self.channels, self.ngf, self.norm_type,
                                               self.init_type, self.init_gain, self.dropout,
                                               self.is_training, n_blocks=6)
            elif self.netG == 'unet_256':
                print("Haven't implemented yet...")
                sys.exit(1)
            elif self.netG == 'unet_128':
                print("Haven't implemented yet...")
                sys.exit(1)
            elif self.netG == 'sid_unet_resize':
                output = self.unet_resize_conv(input, gray, self.channels, self.ngf, self.norm_type,
                                               self.init_type, self.init_gain, self.dropout, self.is_training,
                                               self.self_attention, self.times_residual, self.skip)
            else:
                print("Invalid generator architecture.")
                sys.exit(1)

        # set reuse to True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def resnet_generator(self, input, channels=3, ngf=64, norm_type='instance',
                         init_type='normal', init_gain=1.0, dropout=False,
                         is_training=True, n_blocks=6):
        """
        Resnet-based generator that contains Resnet blocks in between
        some downsampling and upsampling layers.
        """
        def resnet_block(input, channels, filter_size=3, stride=1, norm_type='instance',
                         activation_type='ReLU', is_training=True, dropout=False,
                         scope=None, reuse=False):
            """
            Residual block that contains two 3x3 convolution layers with the same number
            of filters on both layer.
            """
            with tf.variable_scope(scope, reuse=reuse):
                conv1 = ops.conv(input, channels, channels, filter_size=filter_size, stride=stride,
                                 padding_type='REFLECT', norm_type=norm_type, activation_type=activation_type,
                                 is_training=is_training, scope='res_conv1', reuse=reuse)

                if dropout:
                    conv1 = tf.nn.dropout(conv1, keep_prob=0.5)

                conv2 = ops.conv(conv1, channels, channels, filter_size=filter_size, stride=stride,
                                 padding_type='REFLECT', norm_type=norm_type, activation_type=None,
                                 is_training=is_training, scope='res_conv2', reuse=reuse)

                layer = input + conv2

            return layer

        # 7x7 convolution-instance norm-relu layer with 64 filters and stride 1
        c7s1_64 = ops.conv(input, in_channels=channels, out_channels=ngf, filter_size=7, stride=1,
                           padding_type='REFLECT', weight_init_type=init_type, weight_init_gain=init_gain,
                           norm_type=norm_type, is_training=is_training, scope='c7s1-64', reuse=self.reuse)

        # 3x3 convolution-instance norm-relu layer with 128 filters and stride 2
        d128 = ops.conv(c7s1_64, in_channels=ngf, out_channels=2*ngf, filter_size=3,
                        stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                        norm_type=norm_type, is_training=is_training, scope='d128', reuse=self.reuse)

        # 3x3 convolution-instance norm-relu layer with 256 filters and stride 2
        d256 = ops.conv(d128, in_channels=2*ngf, out_channels=4*ngf, filter_size=3,
                        stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                        norm_type=norm_type, is_training=is_training, scope='d256', reuse=self.reuse)

        r256 = d256

        # Resnet blocks with 256 filters
        for idx in range(n_blocks):
            # residual block that contains two 3x3 convolution layers with 256 filters and stride 1
            r256 = resnet_block(r256, channels=4*ngf, filter_size=3, stride=1, norm_type=norm_type,
                                is_training=is_training, dropout=dropout, scope='r256-'+str(idx),
                                reuse=self.reuse)

        # 3x3 fractional strided convolution-instance norm-relu layer with 128 filters and stride 1/2
        u128 = ops.transpose_conv(r256, in_channels=4*ngf, out_channels=2*ngf, filter_size=3,
                                  stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                                  norm_type=norm_type, is_training=is_training, scope='u128', reuse=self.reuse)

        # 3x3 fractional strided convolution-instance norm-relu layer with 64 filters and stride 1/2
        u64 = ops.transpose_conv(u128, in_channels=2*ngf, out_channels=ngf, filter_size=3,
                                 stride=2, weight_init_type=init_type, weight_init_gain=init_gain,
                                 norm_type=norm_type, is_training=is_training, scope='u64', reuse=self.reuse)

        # 7x7 convolution-instance norm-relu layer with 3 filters and stride 1
        c7s1_3 = ops.conv(u64, in_channels=ngf, out_channels=channels, filter_size=7, stride=1,
                          padding_type='REFLECT', weight_init_type=init_type, weight_init_gain=init_gain,
                          use_bias=False, norm_type=None, activation_type=None, is_training=is_training,
                          scope='c7s1-3', reuse=self.reuse)

        return tf.math.tanh(c7s1_3+input, name='gen_out')

    def unet_resize_conv(self, input, gray, channels=3, ngf=64, norm_type='instance',
                         init_type='normal', init_gain=1.0, dropout=False,
                         is_training=True, self_attention=True, times_residual=True, skip=0.0):
        """
        Unet-based generator that contains Resnet blocks in between
        some downsampling and upsampling layers.
        """
        input, pad_left, pad_right, pad_top, pad_bottom = ops.pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = ops.pad_tensor(gray)

        if self_attention:
            gray_2 = ops.max_pooling(gray)
            gray_3 = ops.max_pooling(gray_2)
            gray_4 = ops.max_pooling(gray_3)
            gray_5 = ops.max_pooling(gray_4)

        in_channels = channels+1 if self_attention else channels
        x = tf.concat([input, gray], -1) if self_attention else input

        x = ops.conv(x, in_channels=in_channels, out_channels=ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv1', reuse=self.reuse)

        conv_block1 = ops.conv(x, in_channels=ngf, out_channels=ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv2', reuse=self.reuse)

        x = ops.max_pooling(conv_block1)

        x = ops.conv(x, in_channels=ngf, out_channels=2*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv3', reuse=self.reuse)

        conv_block2 = ops.conv(x, in_channels=2*ngf, out_channels=2*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv4', reuse=self.reuse)

        x = ops.max_pooling(conv_block2)

        x = ops.conv(x, in_channels=2*ngf, out_channels=4*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv5', reuse=self.reuse)

        conv_block3 = ops.conv(x, in_channels=4*ngf, out_channels=4*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv6', reuse=self.reuse)

        x = ops.max_pooling(conv_block3)

        x = ops.conv(x, in_channels=4*ngf, out_channels=8*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv7', reuse=self.reuse)

        conv_block4 = ops.conv(x, in_channels=8*ngf, out_channels=8*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv8', reuse=self.reuse)

        x = ops.max_pooling(conv_block4)

        x = ops.conv(x, in_channels=8*ngf, out_channels=16*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv9', reuse=self.reuse)

        x = x * gray_5 if self_attention else x

        conv_block5 = ops.conv(x, in_channels=16*ngf, out_channels=16*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv10', reuse=self.reuse)

        upsample_block1 = ops.upsample(conv_block5, rescale_factor=2, in_channels=16*ngf, out_channels=8*ngf,
                                       filter_size=3, stride=1, padding_type='SAME', weight_init_type=init_type,
                                       weight_init_gain=init_gain, norm_type=None, activation_type=None,
                                       is_training=is_training, scope='deconv1', reuse=self.reuse)

        x = conv_block4 * gray_4 if self_attention else conv_block4

        x = tf.concat([upsample_block1, x], -1)

        x = ops.conv(x, in_channels=16*ngf, out_channels=8*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv11', reuse=self.reuse)

        conv_block6 = ops.conv(x, in_channels=8*ngf, out_channels=8*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv12', reuse=self.reuse)

        upsample_block2 = ops.upsample(conv_block6, rescale_factor=2, in_channels=8*ngf, out_channels=4*ngf,
                                       filter_size=3, stride=1, padding_type='SAME', weight_init_type=init_type,
                                       weight_init_gain=init_gain, norm_type=None, activation_type=None,
                                       is_training=is_training, scope='deconv2', reuse=self.reuse)

        x = conv_block3 * gray_3 if self_attention else conv_block3

        x = tf.concat([upsample_block2, x], -1)

        x = ops.conv(x, in_channels=8*ngf, out_channels=4*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv13', reuse=self.reuse)

        conv_block7 = ops.conv(x, in_channels=4*ngf, out_channels=4*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv14', reuse=self.reuse)

        upsample_block3 = ops.upsample(conv_block7, rescale_factor=2, in_channels=4*ngf, out_channels=2*ngf,
                                       filter_size=3, stride=1, padding_type='SAME', weight_init_type=init_type,
                                       weight_init_gain=init_gain, norm_type=None, activation_type=None,
                                       is_training=is_training, scope='deconv3', reuse=self.reuse)

        x = conv_block2 * gray_2 if self_attention else conv_block2

        x = tf.concat([upsample_block3, x], -1)

        x = ops.conv(x, in_channels=4*ngf, out_channels=2*ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv15', reuse=self.reuse)

        conv_block8 = ops.conv(x, in_channels=2*ngf, out_channels=2*ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv16', reuse=self.reuse)

        upsample_block4 = ops.upsample(conv_block8, rescale_factor=2, in_channels=2*ngf, out_channels=ngf,
                                       filter_size=3, stride=1, padding_type='SAME', weight_init_type=init_type,
                                       weight_init_gain=init_gain, norm_type=None, activation_type=None,
                                       is_training=is_training, scope='deconv4', reuse=self.reuse)

        x = conv_block1 * gray if self_attention else conv_block1

        x = tf.concat([upsample_block4, x], -1)

        x = ops.conv(x, in_channels=2*ngf, out_channels=ngf, filter_size=3, stride=1,
                     padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                     norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                     scope='conv17', reuse=self.reuse)

        conv_block9 = ops.conv(x, in_channels=ngf, out_channels=ngf, filter_size=3, stride=1,
                               padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                               norm_type=norm_type, activation_type='LeakyReLU', is_training=is_training,
                               scope='conv18', reuse=self.reuse)

        latent = ops.conv(conv_block9, in_channels=ngf, out_channels=3, filter_size=1, stride=1,
                          padding_type='SAME', weight_init_type=init_type, weight_init_gain=init_gain,
                          norm_type=None, activation_type=None, is_training=is_training,
                          scope='conv19', reuse=self.reuse)

        latent = latent * gray if times_residual else latent
        output = latent + input*skip if skip else latent

        output = ops.pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = ops.pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = ops.pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)

        if skip:
            return output, latent
        else:
            return output
