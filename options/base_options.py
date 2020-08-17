import argparse
from abc import ABC, abstractmethod


class BaseOptions(ABC):
    """
    Base arguments for any model.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # dataset arguments
        self.parser.add_argument('--data_loader', type=str, default='unpaired',
                                help='Specify which data loader to use [single | paired | unpaired]')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Input batch size')
        self.parser.add_argument('--channels', type=int, default=3, help='Number of channels of input images')
        self.parser.add_argument('--scale', type=str, default='make_power_2',
                                help='Rescale images [make_power_2 | center_crop | random_crop]')
        self.parser.add_argument('--scale_size', type=int, default=286, help='Resize images to specific size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='Crop images to specific size')
        self.parser.add_argument('--flip', action='store_true', help='Randomly flip images if set')
        self.parser.add_argument('--norm_type', type=str, default='standardize',
                                help='Type of data normalization to perform [standardize | normalize | subtract_mean]')
        self.parser.add_argument('--per_channel_norm', action='store_true', help='Perform per channel normalization if set')
        self.parser.add_argument('--norm_mean', type=float, default=None, help='Value to subtract from images')
        self.parser.add_argument('--norm_min', type=float, default=0., help='Min. possible pixel value')
        self.parser.add_argument('--norm_max', type=float, default=255., help='Max possible pixel value')

        # model arguments
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
        self.parser.add_argument('--dropout', action='store_true', help='Perform dropout if set')
        self.parser.add_argument('--layer_norm_type', type=str, default='batch',
                                help='Type of normalization [batch | instance]')
        self.parser.add_argument('--weight_init_type', type=str, default='normal',
                                help='Type of weight initialization [normal | he | orthogonal]')
        self.parser.add_argument('--weight_init_gain', type=float, default=1.0,
                                help='Scaling factor of weight initialization')

        # training arguments
        self.parser.add_argument('--load_model', type=str, default=None,
                                help='Load a model to continue training where you left off.')
        self.parser.add_argument('--display_frequency', type=int, default=25,
                                help='The number of training steps before printing loss')
        self.parser.add_argument('--checkpoint_frequency', type=int, default=100,
                                help='The number of training steps before saving a checkpoint')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Moment term for adam. Default is 0.5')
        self.parser.add_argument('--niter', type=int, default=100000, help='# of steps at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100000,
                                help='# of steps to linearly decay learning rate to zero')

        # testing arguments
        self.parser.add_argument('--num_samples', type=int, default=32,
                                help='Number of samples you would like to generate.')
        self.parser.add_argument('--sample_directory', type=str, default='./samples/',
                                help='Directory in which samples will be saved to.')

    @abstractmethod
    def parse(self):
        raise NotImplementedError
