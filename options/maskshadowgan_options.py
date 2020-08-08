import argparse
from options.base_options import BaseOptions


class MaskShadowGANOptions(BaseOptions):
    """
    Additional arguments for Mask-ShadowGAN model.
    """
    def __init__(self, training):
        BaseOptions.__init__(self)

        # dataset arguments
        if training:  # assumes unpaired data loader is used during training
            self.parser.add_argument('--dirA', type=str, required=True, help='Path to training shadow dataset')
            self.parser.add_argument('--dirB', type=str, required=True, help='Path to training shadow free dataset')
        else:  # assumes single data loader is used during testing
            self.parser.add_argument('--dir', type=str, required=True, help='Path to test shadow dataset')

        # model arguments
        self.parser.add_argument('--lamA', type=float, default=10.0, help='weight for forward cycle loss (A->B->A)')
        self.parser.add_argument('--lamB', type=float, default=10.0, help='weight for backward cycle loss (B->A->B)')
        self.parser.add_argument('--lambda_ident', type=float, default=0.0, help='weight for identity loss')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of filters in first conv. layer of generator')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of filters in first conv. layer of discriminator')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--queue_size', type=int, default=100,
                                help='the size of mask queue that stores previously generated shadow masks')

    def parse(self):
        return self.parser.parse_args()
