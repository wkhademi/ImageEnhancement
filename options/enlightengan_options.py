import argparse
from options.base_options import BaseOptions


class EnlightenGANOptions(BaseOptions):
    """
    Additional arguments for EnlightenGAN model.
    """
    def __init__(self, training):
        BaseOptions.__init__(self)

        # dataset arguments
        if training:  # assumes unpaired data loader is used during training
            self.parser.add_argument('--dirA', type=str, required=True, help='Path to training dataset A')
            self.parser.add_argument('--dirB', type=str, required=True, help='Path to training dataset B')
        else:  # assumes single data loader is used during testing
            self.parser.add_argument('--dir', type=str, required=True, help='Path to test dataset')

        # generator arguments
        self.parser.add_argument('--ngf', type=int, default=32, help='# of filters in first conv. layer of generator')
        self.parser.add_argument('--netG', type=str, default='sid_unet_resize',
                                help='Specify generator architecture [resnet_9blocks | resnet_6blocks | sid_unet_resize]')
        self.parser.add_argument('--self_attention', action='store_true', help='Adding attention on the input of generator')
        self.parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
        self.parser.add_argument('--skip', type=float, default=1.0, help='B = G(A) + skip*A')

        # discriminator arguments
        self.parser.add_argument('--ndf', type=int, default=64, help='# of filters in first conv. layer of discriminator')
        self.parser.add_argument('--netD', type=str, default='no_norm_n_layers',
                                help='Specify discriminator architecture [basic | n_layers | no_norm_n_layers | pixel].')
        self.parser.add_argument('--n_layers', type=int, default=5,
                                help='# of layers for discriminator. Only used if netD==[n_layers | no_norm_n_layers]')
        self.parser.add_argument('--n_layers_patch', type=int, default=4,
                                help='# of layers for patch discriminator. Only used if netD==[n_layers | no_norm_n_layers]')
        self.parser.add_argument('--patchD', action='store_true', help='Use patch discriminator')
        self.parser.add_argument('--patchD_3', type=int, default=0, help='Choose number of crops for patch discriminator')
        self.parser.add_argument('--patch_size', type=int, default=32, help='Size to crop patches to')

        # other arguments
        self.parser.add_argument('--vgg', action='store_true', help='Use perceptual loss')
        self.parser.add_argument('--vgg_choose', type=str, default='block5_conv1', help='Choose layer of VGG')
        self.parser.add_argument('--no_vgg_instance', action='store_true', help='Whether to apply instance normalization on extracted features')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='Use least square GAN or vanilla GAN. Default is LSGAN.')
        self.parser.add_argument('--use_ragan', action='store_true', help='Use ragan')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='Use lsgan and ragan separately')

    def parse(self):
        return self.parser.parse_args()
