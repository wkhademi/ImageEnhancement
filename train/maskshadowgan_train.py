import os
import sys
import tensorflow as tf
from datetime import datetime
from utils.image_pool import ImagePool
from train.base_train import BaseTrain
from options.maskshadowgan_options import MaskShadowGANOptions


class MaskShadowGANTrain(BaseTrain):
    """
    Trainer for MaskShadowGAN model.
    """
    def __init__(self, opt):
        BaseTrain.__init__(self, opt)

    def train(self):
        """
        Train the MaskShadowGAN model by starting from a saved checkpoint or from
        the beginning.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/' + self.opt.load_model
        else:
            checkpoint_name = datetime.now().strftime("%d%m%Y-%H%M")
            checkpoint = 'checkpoints/{}'.format(checkpoint_name)

            try:
                os.makedirs(checkpoint)
            except os.error:
                print("Failed to make new checkpoint directory.")
                sys.exit(1)

        # TO DO: add graph creation and training loop


if __name__ == '__main__':
    parser = MaskShadowGANOptions(True)
    opt = parser.parse()
    trainer = MaskShadowGANTrain(opt)
    trainer.train()
