import os
import sys
import tensorflow as tf
from test.base_test import BaseTest
from utils.file_utils import save_image
from options.maskshadowgan_options import MaskShadowGANOptions


class MaskShadowGANTest(BaseTest):
    """
    Tester for MaskShadowGAN model.
    """
    def __init__(self, opt):
        BaseTest.__init__(self, opt)

    def test(self):
        """
        Test the MaskShadowGAN model by loading in a saved model.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/' + self.opt.load_model
        else:
            print("Must load in a model to test on.")
            sys.exit(1)

        # TO DO: add graph creation and sample generation


if __name__ == '__main__':
    parser = MaskShadowGANOptions(False)
    opt = parser.parse()
    tester = MaskShadowGANTest(opt)
    tester.test()
