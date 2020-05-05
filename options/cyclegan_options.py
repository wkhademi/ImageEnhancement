import argparse
from options.base_options import BaseOptions


class CycleGANOptions(BaseOptions):
    """
    Additional arguments for CycleGAN model.
    """
    def __init__(self, training):
        BaseOptions.__init__(self)

        # dataset arguments
        if training:  # assumes unpaired data loader is used during training
            self.parser.add_argument('--dirA', type=str, required=True, help='Path to training dataset A')
            self.parser.add_argument('--dirB', type=str, required=True, help='Path to training dataset B')
        else:  # assumes single data loader is used during testing
            self.parser.add_argument('--dir', type=str, required=True, help='Path to test dataset')

        # model arguments
        self.parser.add_argument('--lamA', type=float, default=10.0, help='weight for forward cycle loss (A->B->A)')
        self.parser.add_argument('--lamB', type=float, default=10.0, help='weight for backward cycle loss (B->A->B)')
        self.parser.add_argument('--lambda_ident', type=float, default=0.0, help='weight for identity loss')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of filters in first conv. layer of generator')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of filters in first conv. layer of discriminator')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks',
                                help='Specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.parser.add_argument('--netD', type=str, default='basic',
                                help='Specify discriminator architecture [basic | n_layers | pixel].')
        self.parser.add_argument('--n_layers', type=int, default=3,
                                help='# of layers for discriminator. Only used if netD==n_layers')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='Use least square GAN or vanilla GAN. Default is LSGAN.')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Moment term for adam. Default is 0.5')
        self.parser.add_argument('--niter', type=int, default=100000, help='# of steps at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100000,
                                help='# of steps to linearly decay learning rate to zero')

    def parse(self):
        return self.parser.parse_args()
