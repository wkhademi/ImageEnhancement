import argparse
from options.base_options import BaseOptions


class SRGANOptions(BaseOptions):
    """
    Additional arguments for SRGAN model.
    """
    def __init__(self, training):
        BaseOptions.__init__(self)

        self.parser.add_argument('--dir', type=str, required=True, help='Path to test dataset')
        self.parser.add_argument('--num_epochs_init', type=int, default=100, help='Number of epochs to train generator for initialization')
        self.parser.add_argument('--num_epochs', type=int, default=2000, help='Number of epochs to train for')
        self.parser.add_argument('--lr_decay', type=float, default=0.1, help='Decay learning rate')
        self.parser.add_argument('--decay_every', type=int, default=1000, help='When to decay learning rate')

    def parse(self):
        return self.parser.parse_args()
